import copy
import json
import random
import shutil
import time
from functools import partial
from importlib import resources

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from loguru import logger
from pygments import formatters, highlight, lexers
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from torchllms import distributed as tl_dist
from torchllms import models
from torchllms.messages import Role, configs, tokenization
from torchllms.training import arguments, data, optim, utils


def add_dpo_args(parser):
    parser.add_argument(
        "--dpo_beta",
        type=float,
        default=0.1,
        help="DPO beta parameter",
    )
    parser.add_argument(
        "--dpo_label_smoothing",
        type=float,
        default=0.0,
        help="DPO label smoothing",
    )
    return parser


def format_and_tokenize(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    template_config: tokenization.TemplateConfig,
):
    conversation = tokenization.Conversation(
        messages=example["chosen"], tools=example.get("tools", None)
    )
    input_ids, role_ids = tokenization.tokenize_conversation(
        conversation, tokenizer, template_config
    )
    input_ids_t = torch.as_tensor(input_ids, dtype=torch.int64)
    role_ids_t = torch.as_tensor(role_ids, dtype=torch.int64)
    mask_t = role_ids_t == int(Role.ASSISTANT)  # only train on assistant tokens
    labels_t = input_ids_t.clone().where(mask_t, data.IGNORE_ID)

    example["input_ids"] = input_ids
    example["role_ids"] = role_ids
    example["labels"] = labels_t.tolist()[1:] + [data.IGNORE_ID]

    rejected = tokenization.Conversation(
        messages=example["rejected"], tools=example.get("tools", None)
    )
    rejected_ids, rejected_role_ids = tokenization.tokenize_conversation(
        rejected, tokenizer, template_config
    )
    rejected_ids_t = torch.as_tensor(rejected_ids, dtype=torch.int64)
    rejected_role_ids_t = torch.as_tensor(rejected_role_ids, dtype=torch.int64)
    rejected_mask_t = rejected_role_ids_t == int(Role.ASSISTANT)
    rejected_labels_t = rejected_ids_t.clone().where(rejected_mask_t, data.IGNORE_ID)

    example["input_ids_rejected"] = rejected_ids
    example["role_ids_rejected"] = rejected_role_ids
    example["labels_rejected"] = rejected_labels_t.tolist()[1:] + [data.IGNORE_ID]

    return example


@logger.catch(onerror=lambda e: dist.destroy_process_group())
def main():
    args = add_dpo_args(arguments.get_parser()).parse_args()

    assert not args.pack_samples, "Packing samples not supported for DPO"

    local_rank = tl_dist.setup_distributed()
    device = torch.device("cuda", local_rank)
    global_rank = dist.get_rank()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.opt(colors=True).info(f"<green>Initialized on rank:</green> {global_rank}")

    dist.barrier()

    if global_rank == 0:
        json_str = json.dumps(vars(args), indent=2, cls=utils.JSONEncoder)
        json_str = highlight(
            json_str, lexers.JsonLexer(), formatters.TerminalFormatter()
        )
        logger.opt(colors=True).info(f"<green>Arguments:</green>\n{json_str}")
        logger.opt(colors=True).info(f"PyTorch version: {torch.__version__}")
        logger.opt(colors=True).info("<green>Loading model and tokenizer...</green>")

    main_ckpt_dir = args.ckpt_paths[-1].parent.resolve()
    with open(main_ckpt_dir / "params.json") as f:
        params = json.load(f)

    # update params with args
    params.update(
        {
            "max_seq_len": args.max_seq_len,
            "attention_impl": args.attention_impl,
            "use_role_embeddings": args.use_role_embeddings,
            "role_embeddings_init": args.role_embeddings_init,
        }
    )
    model_params = models.networks.ModelParams(**params)

    with torch.device("meta"):
        model = models.networks.Transformer.from_params(model_params)

    model = models.load_model_weights(
        args.ckpt_paths,
        model=model,
        precision="bfloat16",
        device=device,
    )

    with torch.device("cuda", index=local_rank):
        models.utils.init_meta_params(model)

    if not args.lora:
        ref_model = copy.deepcopy(model)
    else:
        replace_linear_with_lora = models.lora.make_replace_linear_with_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        model.apply(replace_linear_with_lora)
        model.apply(models.lora.freeze_non_lora_params)

        if args.lora_embedding:
            replace_embedding_with_lora = models.lora.make_replace_embedding_with_lora(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
            model.apply(replace_embedding_with_lora)
            model.apply(models.lora.freeze_non_lora_params)

        models.lora.unfreeze_params(model, args.lora_unfrozen_params)

    # requires_grad is reset when transferring to device
    trainable_params = []
    frozen_params = []
    num_trainable = 0
    num_frozen = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            num_trainable += p.numel()
            trainable_params.append(n)
        else:
            num_frozen += p.numel()
            frozen_params.append(n)

    if global_rank == 0:
        trainable_params_str = "\n".join(
            models.utils.collate_param_names(trainable_params)
        )
        logger.opt(colors=True).info(
            f"<yellow>Trainable params ({num_trainable // 1e6}M):</yellow>\n<dim>{trainable_params_str}</dim>"
        )

        frozen_params_str = "\n".join(models.utils.collate_param_names(frozen_params))
        logger.opt(colors=True).info(
            f"<yellow>Frozen params ({num_frozen // 1e6}M):</yellow>\n<dim>{frozen_params_str}</dim>"
        )

    tl_dist.wrap_fsdp(model)
    if not args.lora:
        tl_dist.wrap_fsdp(ref_model)

    if args.selective_ac_ratio > 0:
        utils.apply_fsdp_checkpointing(
            model, block=model.block_cls, p=args.selective_ac_ratio
        )

    tokenizer = AutoTokenizer.from_pretrained(main_ckpt_dir)
    config_path = resources.files(configs).joinpath(args.template_config)
    with config_path.open() as file:
        template_config = tokenization.TemplateConfig(**yaml.safe_load(file))

    if args.output_dir is not None:
        config_path_json = config_path.with_suffix(".json")
        target_path = args.output_dir / "tokenizer_config.json"
        shutil.copy(str(config_path_json.resolve()), str(target_path.resolve()))

    # Prepare the data
    if global_rank == 0:
        logger.opt(colors=True).info(
            "<green>Loading and tokenizing the dataset...</green>"
        )

    format_and_tokenize_fn = partial(
        format_and_tokenize,
        tokenizer=tokenizer,
        template_config=template_config,
    )
    train_dataset = data.create_dataset(
        args,
        tokenizer,
        format_and_tokenize_fn,
        columns=[
            "input_ids",
            "input_ids_rejected",
            "role_ids",
            "role_ids_rejected",
            "labels",
            "labels_rejected",
        ],
    )

    batch_size_per_gpu = args.micro_batch_size_per_gpu * args.gradient_accum_steps
    global_batch_size = batch_size_per_gpu * dist.get_world_size()

    args.train_steps = args.train_epochs * (len(train_dataset) // global_batch_size)
    num_samples = args.train_steps // args.train_epochs * global_batch_size

    if global_rank == 0:
        logger.opt(colors=True).info(
            f"<yellow>Setting train steps based on specified epochs:</yellow> {args.train_steps}"
        )
        logger.opt(colors=True).info(
            f"<green>Trimming to fit global batch size:</green> {len(train_dataset) - num_samples} samples dropped"
        )

    train_dataset = train_dataset.select(range(num_samples))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=DistributedSampler(train_dataset, drop_last=True, seed=args.seed),
        collate_fn=data.DataCollator(
            keys=[
                "input_ids",
                "input_ids_rejected",
                "role_ids",
                "role_ids_rejected",
                "labels",
                "labels_rejected",
            ],
            pad_ids=[
                tokenizer.eos_token_id,
                tokenizer.eos_token_id,
                int(Role.OTHER),
                int(Role.OTHER),
                data.IGNORE_ID,
                data.IGNORE_ID,
            ],
            max_seq_len=args.max_seq_len,
        ),
    )

    # Create optimizer and lr scheduler
    optimizer = optim.create_optimizer(args, model)
    scheduler = optim.create_scheduler(args, optimizer, args.train_steps)
    args.clip_grad_norm = (
        float(args.clip_grad_norm) if args.clip_grad_norm > 0 else float("inf")
    )

    if global_rank == 0 and args.wandb:
        import wandb

        wandb.require("core")

        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            config=args,
        )

    # Snapshot the script arguments and copy the tokenizer
    if global_rank == 0 and args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_args_and_params(args, params, main_ckpt_dir)
        tokenizer.save_pretrained(args.output_dir)

    model.train()
    utils.cleanup_memory()

    global_step = 0
    global_step_time_ema = None
    global_tokens_ema = None
    global_train_tokens_ema = None

    if global_rank == 0:
        logger.opt(colors=True).info(
            f"<green>Training for {args.train_steps} steps...</green>",
        )

    for epoch in range(args.train_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            global_step += 1
            start = time.time()

            batch = tl_dist.to_device(batch, device)
            micro_batches, micro_tokens, micro_train_tokens = data.get_micro_batches(
                batch, args
            )

            global_loss = 0
            global_accuracy = 0
            global_tokens = sum(micro_tokens)
            global_train_tokens = sum(micro_train_tokens)

            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_train_tokens, op=dist.ReduceOp.SUM)

            for micro_batch in micro_batches:
                input_ids_w = micro_batch["input_ids"]
                input_ids_l = micro_batch["input_ids_rejected"]
                role_ids_w = micro_batch["role_ids"]
                role_ids_l = micro_batch["role_ids_rejected"]
                labels_w = micro_batch["labels"].unsqueeze(2)
                labels_l = micro_batch["labels_rejected"].unsqueeze(2)

                mask_w = labels_w != data.IGNORE_ID
                mask_l = labels_l != data.IGNORE_ID

                labels_w = torch.where(mask_w, labels_w, 0)
                labels_l = torch.where(mask_l, labels_l, 0)

                logits_w, _ = model(input_ids_w, role_ids_w)
                logits_l, _ = model(input_ids_l, role_ids_l)
                if args.fp32_logits:
                    logits_w = logits_w.float()
                    logits_l = logits_l.float()
                token_logprobs_w = F.log_softmax(logits_w, dim=-1)
                token_logprobs_l = F.log_softmax(logits_l, dim=-1)
                token_logprobs_w = torch.gather(token_logprobs_w, 2, labels_w)
                token_logprobs_l = torch.gather(token_logprobs_l, 2, labels_l)

                logprobs_w = (token_logprobs_w * mask_w).sum((1, 2))
                logprobs_l = (token_logprobs_l * mask_l).sum((1, 2))

                del logits_w, logits_l

                with torch.no_grad():
                    if not args.lora:
                        ref_logits_w, _ = ref_model(input_ids_w)
                        ref_logits_l, _ = ref_model(input_ids_l)
                    else:
                        with models.lora.disable_lora(model):
                            ref_logits_w, _ = model(input_ids_w)
                            ref_logits_l, _ = model(input_ids_l)
                    if args.fp32_logits:
                        ref_logits_w = ref_logits_w.float()
                        ref_logits_l = ref_logits_l.float()
                    ref_token_logprobs_w = F.log_softmax(ref_logits_w, dim=-1)
                    ref_token_logprobs_l = F.log_softmax(ref_logits_l, dim=-1)
                    ref_token_logprobs_w = torch.gather(
                        ref_token_logprobs_w, 2, labels_w
                    )
                    ref_token_logprobs_l = torch.gather(
                        ref_token_logprobs_l, 2, labels_l
                    )
                    ref_logprobs_w = (ref_token_logprobs_w * mask_w).sum((1, 2))
                    ref_logprobs_l = (ref_token_logprobs_l * mask_l).sum((1, 2))

                del ref_logits_w, ref_logits_l

                logratios = logprobs_w - logprobs_l
                ref_logratios = ref_logprobs_w - ref_logprobs_l

                logits = logratios - ref_logratios

                loss = (
                    -F.logsigmoid(args.dpo_beta * logits)
                    * (1 - args.dpo_label_smoothing)
                    - F.logsigmoid(-args.dpo_beta * logits) * args.dpo_label_smoothing
                ).mean()

                chosen_rewards = args.dpo_beta * (logprobs_w - ref_logprobs_w).detach()
                rejected_rewards = (
                    args.dpo_beta * (logprobs_l - ref_logprobs_l).detach()
                )
                global_accuracy += (
                    chosen_rewards > rejected_rewards
                ).float().mean() / args.gradient_accum_steps

                if loss.isnan():
                    raise Exception("Loss is NaN, training diverged!")

                loss /= args.gradient_accum_steps
                loss.backward()
                global_loss += loss.detach()

            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(global_accuracy, op=dist.ReduceOp.AVG)

            # noop if max_norm=inf
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.clip_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step_time = time.time() - start

            if global_step_time_ema is None:
                global_step_time_ema = global_step_time

            if global_tokens_ema is None:
                global_tokens_ema = global_tokens

            if global_train_tokens_ema is None:
                global_train_tokens_ema = global_train_tokens

            # Compute logging metrics
            global_step_time_ema = 0.9 * global_step_time_ema + 0.1 * global_step_time
            global_tokens_ema = 0.9 * global_tokens_ema + 0.1 * global_tokens
            global_train_tokens_ema = (
                0.9 * global_train_tokens_ema + 0.1 * global_train_tokens
            )
            global_tokens_per_sec_ema = global_tokens_ema / global_step_time_ema
            global_train_tokens_per_sec_ema = (
                global_train_tokens_ema / global_step_time_ema
            )
            train_rate_ema = global_batch_size / global_step_time_ema
            current_lr = optimizer.param_groups[0]["lr"]
            max_mem_allocated = torch.cuda.max_memory_allocated() / 1e9
            max_mem_reserved = torch.cuda.max_memory_reserved() / 1e9

            if global_rank == 0 and global_step % args.print_freq == 0:
                eta = (args.train_steps - global_step) * global_step_time_ema

                logger.opt(colors=True).info(
                    f"<green>[global step {global_step}/{args.train_steps}]</green> "
                    f"time={global_step_time:.2f}s ({eta:.0f}s left) | "
                    f"loss={global_loss.item():.3f} "
                    f"grad_norm={total_norm.item():.3f} "
                    f"accuracy={global_accuracy.item():.3f} "
                    f"lr={current_lr:.2e} "
                    f"train_tok={global_train_tokens.item()} "
                    f"total_tok={global_tokens.item()} | "
                    f"global_ex/s={train_rate_ema:.2f} "
                    f"global_train_tok/s={global_train_tokens_per_sec_ema.item():.1f} "
                    f"global_total_tok/s={global_tokens_per_sec_ema.item():.1f} "
                    f"max_mem_allocated={max_mem_allocated:.1f}GB "
                    f"max_mem_reserved={max_mem_reserved:.1f}GB"
                )

            if global_rank == 0 and args.wandb:
                wandb.log(
                    {
                        "loss": global_loss.item(),
                        "accuracy": global_accuracy.item(),
                        "tokens": global_tokens.item(),
                        "train_tokens": global_train_tokens.item(),
                        "lr": current_lr,
                        "grad_norm": total_norm.item(),
                        "global_step_time": global_step_time,
                    },
                    step=global_step,
                )

            if global_step % args.save_freq == 0:
                start = time.time()
                utils.save_model(
                    model,
                    args.output_dir,
                    step_name=f"{global_step:07d}",
                    max_checkpoints=args.max_checkpoints,
                )
                save_time = time.time() - start
                if global_rank == 0:
                    logger.opt(colors=True).info(
                        f"<yellow>Saved checkpoint in {save_time:.1f}s</yellow>"
                    )

            if global_step >= args.train_steps:
                break

    if global_rank == 0:
        logger.opt(colors=True).info("<green>Finished training!</green>")
        logger.opt(colors=True).info(
            "<green>Comparing final weights to initial checkpoints...</green>"
        )

    dist.barrier()

    utils.check_weights(args.ckpt_paths, model)

    utils.save_model(
        model,
        args.output_dir,
        step_name="final",
        max_checkpoints=args.max_checkpoints,
        frozen_params=frozen_params,
    )


if __name__ == "__main__":
    main()
    dist.destroy_process_group()
