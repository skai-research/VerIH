import os
import json
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
from transformers import AutoTokenizer
import argparse
import random
random.seed(42)


def make_prefix(dp, split, tokenizer):
    cot_prompt = '''To follow the user prompt within the constraints of the system prompt, you need to think step by step about the relationship between the system prompt and the user prompt.
Check whether there is any logical conflict between them. Output a single word: 'true' if they are aligned, 'false' if there is a conflict.
Here is the system prompt: '''
    messages = [
        {"role": "system", "content": cot_prompt + dp['premises']},
        {"role": "user", "content": dp['conclusion']}
    ]
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return prefix


def load_samples(file_list):
    samples = []
    for i in file_list:
        samples.extend(json.load(open(i, encoding='utf-8')))
        print(f"{i}: {len(samples)}")
    random.shuffle(samples)
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, choices=['qwen3', 'phi-4-mini'])
    args = parser.parse_args()

    if args.local_dir is None:
        args.local_dir=f'dataset/{args.template_type}-logic-v0/'

    # Read samples
    train_dataset = load_samples(["dataset/logic/lsat_train.json", "dataset/logic/folio_train.json"])
    test_dataset = load_samples(["dataset/logic/lsat_test.json", "dataset/logic/folio_test.json"])

    # Filter invalid samples
    target_keys = {"premises", "conclusion", "label", "type"}
    train_dataset = [d for d in train_dataset if set(d.keys()) == target_keys]
    test_dataset = [d for d in test_dataset if set(d.keys()) == target_keys]

    # Construct dataset
    if args.template_type == "qwen3":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    elif args.template_type == "phi-4-mini":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-reasoning", trust_remote_code=True)
    else: raise NotImplementedError(f"Not supported template_type {args.template_type}")

    data_source = 'ih'
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, split, tokenizer)

            solution = {
                "label": example['label'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{"content": question}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'type': example["type"]
                }
            }
            return data
        return process_fn

    train_dataset = Dataset.from_list(train_dataset).map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset[0].keys())
    test_dataset = Dataset.from_list(test_dataset).map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset[0].keys())

    # Write dataset
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
