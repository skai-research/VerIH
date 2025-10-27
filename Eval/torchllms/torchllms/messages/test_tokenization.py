"""Test tokenization implementation against HuggingFace."""
import argparse

import yaml
from transformers import AutoTokenizer

from torchllms.messages import tokenization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to model directory containing tokenizer files",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="torchllms/messages/configs/llama3_instruct.jinja",
        help="Path to Jinja2 chat template file",
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        default="torchllms/messages/configs/llama3_instruct.yaml",
        help="Path to YAML template config file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    with open(args.chat_template) as f:
        tokenizer.chat_template = f.read()

    with open(args.yaml_config) as f:
        template_config = tokenization.TemplateConfig(**yaml.safe_load(f))

    test_cases = [
        # Basic conversation
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        # With system message
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing great, thanks!"},
        ],
        # With tool message
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "I need to use a tool to answer that."},
            {"role": "tool", "content": "calculator(2+2) = 4"},
            {"role": "assistant", "content": "The answer is 4"},
        ],
    ]

    for i, messages in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print("Messages:", messages)

        # HuggingFace tokenization
        hf_text = tokenizer.apply_chat_template(messages, tokenize=False)
        hf_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

        # Our tokenization
        our_tokens, role_ids = tokenization.tokenize_conversation(
            messages, tokenizer, template_config
        )
        our_text = tokenizer.decode(our_tokens)

        # Compare results
        print("\nHuggingFace encoded text:")
        print(hf_text)
        print("\nOur encoded text:")
        print(our_text)
        print("\nTexts match:", our_text == hf_text)
        print("Tokens match:", our_tokens == hf_tokens)


if __name__ == "__main__":
    main()