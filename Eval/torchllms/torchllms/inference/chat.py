import argparse
from pathlib import Path

from blessed import Terminal

from torchllms.inference import LLM

term = Terminal()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for TorchLLMs models."
    )
    parser.add_argument(
        "--ckpt_paths",
        nargs="*",
        type=Path,
        default=["/data/norman_mu/models/torchllms/llama-2-7b-chat/checkpoint.00.pth"],
        help="Model weight paths to be loaded sequentially (e.g. the result of partial fine-tuning). Model config should reside in the same directory as the first checkpoint.",
    )
    parser.add_argument(
        "--template_config",
        type=str,
        default=None,
        help="Template config file name in torchllms/messages/configs.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for sampling."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision to use, otherwise default to checkpoint's precision",
    )
    parser.add_argument(
        "--model_kwargs",
        type=dict,
        default={},
        help="Additional kwargs to pass to model initialization",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print("Loading model and tokenizer...")
    llm = LLM(
        ckpt_paths=args.ckpt_paths,
        template_config=args.template_config,
        device=args.device,
        precision=args.precision,
        model_kwargs=args.model_kwargs,
    )

    conversation = []
    system_message = input(term.yellow("[system]: ")).strip()
    if len(system_message) > 0:
        conversation.append({"role": "system", "content": system_message})

    while True:
        content = input(term.yellow("[user]: ")).strip()
        conversation.append({"role": "user", "content": content})

        if content == "!!reset":
            conversation = []
            system_message = input(term.yellow("[system]: ")).strip()
            if len(system_message) > 0:
                conversation.append({"role": "system", "content": system_message})
            continue

        responses = llm.generate(
            conversations=[conversation],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        response = responses[0][-1]["content"]
        print(term.yellow("[assistant]: ") + term.normal + response)


if __name__ == "__main__":
    main()
