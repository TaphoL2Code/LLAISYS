"""Interactive CLI chatbot for LLaiSys Qwen2 model."""

import argparse
import sys
import io
from typing import List, Dict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="LLaiSys Interactive Chatbot")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--max-tokens", default=256, type=int, help="Max new tokens per response")
    parser.add_argument("--temperature", default=0.8, type=float, help="Sampling temperature")
    parser.add_argument("--top-p", default=1.0, type=float, help="Top-p (nucleus) sampling")
    parser.add_argument("--top-k", default=0, type=int, help="Top-k sampling")
    args = parser.parse_args()

    # Load model
    import llaisys
    from transformers import AutoTokenizer
    from llaisys.libllaisys import DeviceType

    device = DeviceType.CPU if args.device == "cpu" else DeviceType.NVIDIA
    print(f"Loading model from {args.model} on {args.device}...")
    model = llaisys.models.Qwen2(args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded. Starting chat...\n")

    messages: List[Dict[str, str]] = []

    print("=" * 60)
    print("  LLaiSys Chatbot")
    print("  Type '/exit' to quit, '/clear' to reset conversation")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages = []
            model.reset_kv()
            print("[Conversation cleared]\n")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response
        print("Assistant: ", end="", flush=True)
        full_response = ""
        for text in model.chat(
            messages,
            tokenizer,
            max_new_tokens=args.max_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            stream=True,
        ):
            new_text = text[len(full_response):]
            full_response = text
            print(new_text, end="", flush=True)
        print("\n")

        # Add assistant response to history
        # Extract only the assistant's response (not the full prompt)
        assistant_text = full_response
        # Find the last assistant response in the decoded text
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()