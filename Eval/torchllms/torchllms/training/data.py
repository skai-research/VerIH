import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import datasets
import jsonlines
import plotext as plt
import torch
from loguru import logger
from transformers import PreTrainedTokenizer

from torchllms.distributed import get_rank
from torchllms.messages.tokenization import Role

IGNORE_ID = -100


def load_dataset(path, split, samples):
    if path.startswith("jsonl:"):
        with jsonlines.open(path[len("jsonl:") :]) as reader:
            data = list(reader)
        ds = datasets.Dataset.from_list(data)
    elif path.startswith("text:"):
        text_path = path[len("text:"):]
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
            data = [{"text": line} for line in lines]
        ds = datasets.Dataset.from_list(data)
    else:
        ds = datasets.load_dataset(path, split=split)

    if samples > 0:
        samples = int(samples) if samples > 1 else int(len(ds) * samples)
        ds = ds.shuffle(seed=42).select(range(samples))
    return ds


def truncate_to_len(example, max_seq_len, columns):
    for col in example:
        if any(re.match(pattern, col) for pattern in columns):
            if isinstance(example[col], list):
                example[col] = example[col][:max_seq_len]
    return example


def format_and_tokenize_text(
    example: dict,
    tokenizer: PreTrainedTokenizer,
):
    input_ids = tokenizer.encode(example["text"], add_special_tokens=False)
    input_ids = input_ids + [tokenizer.eos_token_id]
    role_ids = [int(Role.OTHER)] * len(input_ids)
    
    example["input_ids"] = input_ids
    example["role_ids"] = role_ids
    example["labels"] = input_ids[1:] + [IGNORE_ID]
    return example

    # input_text, target = example["text"].split("####")
    # input_ids = tokenizer.encode(input_text + "####", add_special_tokens=False)
    # target_ids = tokenizer.encode(target, add_special_tokens=False)

    # combined_ids = input_ids + target_ids + [tokenizer.eos_token_id]
    # labels = [IGNORE_ID] * len(input_ids) + target_ids + [tokenizer.eos_token_id]
    # role_ids = [int(Role.OTHER)] * len(combined_ids)
    
    # example["input_ids"] = combined_ids
    # example["role_ids"] = role_ids
    # example["labels"] = labels
    # return example


def create_dataset(
    args,
    tokenizer: PreTrainedTokenizer,
    format_and_tokenize_fn: Callable,
    pack_separators: Optional[dict] = None,
    is_train: bool = True,
    columns: list = ["input_ids", "labels", "role_ids"],
):
    """
    Create dataset from data paths and splits. DistributedSampler handles shuffling.

    Args:
        args: arguments from argparse
        format_and_tokenize_fn: function to format and tokenize data
        pack_separators: dictionary of separators for packing samples.
            Should use eos_id for input_ids and IGNORE_ID for labels
        is_train: whether the dataset is for training
    """
    if "id" not in columns:
        columns.append("id")

    data_paths = args.train_data_paths if is_train else args.val_data_paths
    data_splits = args.train_data_splits if is_train else args.val_data_splits
    num_samples = args.train_num_samples if is_train else args.val_num_samples

    if pack_separators is None:
        pack_separators = {
            "input_ids": tokenizer.eos_token_id,
            "role_ids": int(Role.OTHER),
            "labels": IGNORE_ID,
        }

    if len(data_splits) == 0:
        data_splits = [None] * len(data_paths)

    if len(num_samples) == 0:
        num_samples = [-1] * len(data_paths)

    assert len(data_paths) == len(data_splits)
    assert len(data_paths) == len(num_samples)

    all_datasets = []
    for path, split, samples in zip(data_paths, data_splits, num_samples):
        ds = load_dataset(path, split=split, samples=samples)

        def _preprocess(elem, i):
            if path.startswith("text:"):
                elem = format_and_tokenize_text(elem, tokenizer)
            else:
                elem = format_and_tokenize_fn(elem)
            elem = truncate_to_len(elem, args.max_seq_len, columns)
            elem["id"] = f"{path}-{i}"
            return elem

        ds = ds.map(
            _preprocess,
            num_proc=args.num_data_workers,
            with_indices=True,
        )

        _ds_len = ds.filter(
            lambda x: len(x["input_ids"]) == args.max_seq_len,
            num_proc=args.num_data_workers,
        )
        num_truncated = len(_ds_len)
        if get_rank() == 0:
            logger.opt(colors=True).warning(
                f"<red>{num_truncated} out of {len(ds)} samples at max_seq_len</red>",
            )

        if not args.no_visualize_dataset and get_rank() == 0:
            logger.opt(colors=True).info(f"<green>Visualizing dataset:</green> {path}")
            visualize_dataset(ds, tokenizer)

        ds = ds.remove_columns(
            [
                k
                for k in ds.column_names
                if not any(re.match(pattern, k) for pattern in columns)
            ]
        )
        all_datasets.append(ds)

    dataset = datasets.concatenate_datasets(all_datasets)

    count = len(dataset)
    dataset = dataset.filter(
        lambda x: any(label != IGNORE_ID for label in x["labels"]),
        num_proc=args.num_data_workers,
    )

    if get_rank() == 0:
        logger.opt(colors=True).warning(
            f"<red>Removed {count - len(dataset)} samples with all tokens masked (after truncation)</red>",
        )

    return dataset


def visualize_dataset(dataset, tokenizer):
    """
    Visualize training samples as fed into the model. Tokenizer should have add_prefix_space=False
    """

    example_lens = [len(example["input_ids"]) for example in dataset]
    plt.hist(example_lens)
    plt.title("Sequence lengths")
    plt.xlabel("# tokens")
    plt.ylabel("Occurrences")
    plt.show()

    if get_rank() == 0:
        logger.info("Sampling 10 training examples at random")

    args = []
    format_str = ""
    for i in torch.randperm(len(dataset))[:10].tolist():
        example = dataset[i]
        format_str += "\n<yellow>%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%</yellow>\n"

        prev_label = None
        arg_str = ""
        for tok, cur_label in zip(example["input_ids"], example["labels"]):
            if cur_label != prev_label:
                if prev_label is not None:
                    format_str += "{}</>"
                    args.append(arg_str)
                    arg_str = ""

                prev_label = cur_label
                format_str += "<red>" if cur_label == IGNORE_ID else "<green>"

            arg_str += tokenizer.decode(tok)

        format_str += "{}</>"
        args.append(arg_str)

    if get_rank() == 0:
        try:
            logger.opt(colors=True).info(format_str, *args)
        except Exception as e:
            print(f"Error printing formatted sample:\n{format_str}\n{args}", flush=True)
            logger.opt(colors=True).error(f"Error visualizing dataset: {e}")


# TODO: revisit single sequence packing
def pack_samples(
    dataset,
    max_seq_len,
    separators,
):
    """
    Pack samples into sequences of max length max_seq_len.
    Avoids splitting examples across multiple sequences, truncates examples that are too long, and
    removes columns that are not in separators.

    Args:
        dataset: dataset to pack
        max_seq_len: maximum length of sequences to pack
        separators: dictionary of separators for packing samples.
            Should use eos_id for input_ids and IGNORE_ID for labels
    """
    assert "input_ids" in separators.keys()
    examples = []
    more_examples = True
    iterator = iter(dataset)
    next_example = next(iterator)
    for k in separators.keys():
        next_example[k] = next_example[k][: (max_seq_len - 1)]

    while more_examples:
        example = defaultdict(list)

        while (
            len(example["input_ids"]) + len(next_example["input_ids"]) + 1
            <= max_seq_len
        ):
            for k, s in separators.items():
                example[k].extend(next_example[k] + [s])

            try:
                next_example = next(iterator)
                for k in separators.keys():
                    next_example[k] = next_example[k][: (max_seq_len - 1)]
            except StopIteration:
                more_examples = False
                break

        if len(example["input_ids"]) > 0:
            examples.append(example)

        if len(examples) % 100 == 0 and get_rank() == 0:
            logger.info("Packed %d samples", len(examples))

    examples = datasets.Dataset.from_list(examples)

    if get_rank() == 0:
        logger.opt(colors=True).info(
            f"<yellow>Packed {len(dataset)} samples into {len(examples)} samples</yellow>",
        )

    return examples


@dataclass
class DataCollator(object):
    """
    Collates lists of samples into batched tensors. Handles padding and truncation.
    Input-type tensors are padded with eos_id, label-type tensors are padded with IGNORE_ID.
    """

    keys: list
    pad_ids: list
    max_seq_len: int

    def __call__(self, instances):
        data = {}

        for key, pad_id in zip(self.keys, self.pad_ids):
            data[key] = self._pad(instances, key, pad_value=pad_id)

        data["id"] = [instance["id"] for instance in instances]

        return data

    def _pad(self, instances, key, pad_value):
        if isinstance(instances[0][key], int) or isinstance(instances[0][key], float):
            return torch.as_tensor([instance[key] for instance in instances])

        elems = [
            torch.as_tensor(instance[key][: self.max_seq_len]) for instance in instances
        ]
        out = torch.nn.utils.rnn.pad_sequence(
            elems,
            batch_first=True,
            padding_value=pad_value,
        )
        return out


def get_micro_batches(batch, args):
    mbs = args.micro_batch_size_per_gpu

    micro_batches = []
    micro_tokens = []
    micro_train_tokens = []

    for i in range(args.gradient_accum_steps):
        batch_ = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) or isinstance(v, list):
                batch_[k] = v[i * mbs : (i + 1) * mbs]
            else:
                batch_[k] = v

        micro_batches.append(batch_)

        num_tokens = torch.tensor(
            batch_["labels"].numel(), device=batch_["labels"].device
        )
        micro_tokens.append(num_tokens)

        micro_train_tokens.append((batch_["labels"] != IGNORE_ID).sum())

    return micro_batches, micro_tokens, micro_train_tokens
