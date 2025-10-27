import argparse

import torch
from blessed import Terminal
from transformers import AutoModelForCausalLM, AutoTokenizer

term = Terminal()


def parse_args():
    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/norman_mu/models/Llama-2-7b-chat-hf",
        help="Path to model weights and config",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="Maximum sequence length."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
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
    args = parser.parse_args()
    return args


def main():
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    args = parse_args()

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, args.precision) if args.precision else "auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    conversation = []
    system_message = input(term.yellow("[system]: ")).strip()
    if len(system_message) > 0:
        conversation.append({"role": "system", "content": system_message})

    while True:
        content = input(term.yellow("[user]: ")).strip()
        if content == "!!reset":
            conversation = []
            system_message = input(term.yellow("[system]: ")).strip()
            if len(system_message) > 0:
                conversation.append({"role": "system", "content": system_message})
            continue

        conversation.append({"role": "user", "content": content})
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        encoded = torch.tensor(prompt, dtype=torch.int, device=args.device)[None]

        new_tokens = model.generate(
            input_ids=encoded,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )

        new_tokens = new_tokens[0].tolist()[len(prompt) :]
        new_tokens = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        conversation.append({"role": "assistant", "content": new_tokens})
        print(term.yellow("[assistant]: ") + new_tokens)


if __name__ == "__main__":
    main()
