import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR


def create_optimizer(args, model):
    wd_params, no_wd_params = model.get_wd_params()
    optimizer_grouped_parameters = [
        {
            "params": wd_params,
            "weight_decay": args.wd,
        },
        {
            "params": no_wd_params,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.wd,
        fused=True,
    )
    return optimizer


def create_warmup_scheduler(optimizer, warmup_steps):
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / warmup_steps, 1.0),
    )


def create_linear_scheduler(optimizer, total_steps):
    return LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0,
        total_iters=total_steps,
    )


def create_cosine_scheduler(optimizer, total_steps):
    return CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )


def create_constant_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def create_scheduler(args, optimizer, global_steps):
    if args.warmup_steps > 0:
        warmup_scheduler = create_warmup_scheduler(optimizer, args.warmup_steps)

    if args.warmup_steps > global_steps:
        return warmup_scheduler

    if args.lr_scheduler == "linear":
        scheduler = create_linear_scheduler(optimizer, global_steps - args.warmup_steps)
    elif args.lr_scheduler == "constant":
        scheduler = create_constant_scheduler(optimizer)
    elif args.lr_scheduler == "cosine":
        scheduler = create_cosine_scheduler(optimizer, global_steps - args.warmup_steps)
    else:
        raise ValueError(f"Unknown lr scheduler type: {args.lr_scheduler}")

    if args.warmup_steps > 0:
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[args.warmup_steps],
        )

    return scheduler
