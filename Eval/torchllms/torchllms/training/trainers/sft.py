import json
import random
import shutil
import time
from functools import partial
from importlib import resources

import torch
import torch.distributed as dist
import yaml
from loguru import logger
from pygments import formatters, highlight, lexers
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from torchllms import distributed as tl_dist
from torchllms import models
from torchllms.messages import Role, configs, tokenization
from torchllms.training import arguments, data, losses, optim, utils


def format_and_tokenize(
    example: dict,
    tokenizer: PreTrainedTokenizer,
    template_config: tokenization.TemplateConfig,
):
    conversation = tokenization.Conversation(
        messages=example["messages"], tools=example.get("tools", None)
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
    return example


def validate(model, val_dataloader, loss_fn, device):
    model.eval()
    total_loss = torch.zeros(1, dtype=torch.float32, device=device)
    total_samples = torch.zeros(1, dtype=torch.int64, device=device)
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = tl_dist.to_device(batch, device)
            logits, _ = model(batch["input_ids"], batch["role_ids"])
            loss = loss_fn(logits, batch["labels"])
            
            # Accumulate in float32 to avoid overflow
            total_loss += loss.to(dtype=torch.float32) * len(batch["input_ids"])
            total_samples += len(batch["input_ids"])
    
    # All-reduce to get global stats
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    model.train()
    return (total_loss / total_samples).item()


@logger.catch(onerror=lambda e: dist.destroy_process_group())
def main():
    args = arguments.get_parser().parse_args()

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

    loss_fn = losses.make_loss_fn(args)

    if args.lora:
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

    if args.selective_ac_ratio > 0:
        utils.apply_fsdp_checkpointing(
            model, block=model.block_cls, p=args.selective_ac_ratio
        )

    tokenizer = AutoTokenizer.from_pretrained(main_ckpt_dir)
    template_config = None
    if args.template_config:
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
            keys=["input_ids", "role_ids", "labels"],
            pad_ids=[
                tokenizer.eos_token_id,
                int(Role.OTHER),
                data.IGNORE_ID,
            ],
            max_seq_len=args.max_seq_len,
        ),
    )

    val_dataloader = None
    if args.val_data_paths:
        if global_rank == 0:
            logger.opt(colors=True).info(
                "<green>Loading and tokenizing the validation dataset...</green>"
            )
        
        val_dataset = data.create_dataset(
            args,
            tokenizer,
            format_and_tokenize_fn,
            is_train=False,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.micro_batch_size_per_gpu,
            sampler=DistributedSampler(val_dataset, drop_last=False),
            collate_fn=data.DataCollator(
                keys=["input_ids", "role_ids", "labels"],
                pad_ids=[
                    tokenizer.eos_token_id,
                    int(Role.OTHER),
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
            global_tokens = sum(micro_tokens)
            global_train_tokens = sum(micro_train_tokens)

            dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_train_tokens, op=dist.ReduceOp.SUM)

            if args.loss_reduction == "tokens":
                # Need to re-weight loss b/c num tokens varies across micro-batches and GPUs
                # FSDP collects gradients by avg, autograd collects gradients by sum
                # loss_fn returns 1 / t * sum(per_token_losses)
                loss_scales = [
                    dist.get_world_size() * t / global_train_tokens
                    for t in micro_train_tokens
                ]
            else:
                loss_scales = [1 / args.gradient_accum_steps] * len(micro_batches)

            for micro_batch, loss_scale in zip(micro_batches, loss_scales):
                logits, _ = model(micro_batch["input_ids"], micro_batch["role_ids"])
                loss = loss_fn(logits, micro_batch["labels"])

                if loss.isnan():
                    raise Exception("Loss is NaN, training diverged!")

                del logits

                loss *= loss_scale
                loss.backward()
                global_loss += loss.detach()

            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)

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

            if val_dataloader is not None and (global_step + 1) % args.val_freq == 0:
                val_loss = validate(model, val_dataloader, loss_fn, device)
                if global_rank == 0:
                    logger.opt(colors=True).info(
                        f"<blue>[Validation] Step {global_step}</blue> "
                        f"loss={val_loss:.3f}"
                    )
                if args.wandb and global_rank == 0:
                    wandb.log({"val_loss": val_loss}, step=global_step)

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
