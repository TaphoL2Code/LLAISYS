"""Chat format / template handling for Qwen2 model."""

from typing import List, Dict


def format_chat_messages(messages: List[Dict[str, str]]) -> str:
    """Format a list of chat messages into the Qwen2 chat template format.

    The Qwen2 chat template uses:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>

    Args:
        messages: List of dicts with 'role' and 'content' keys.
                  Roles: 'system', 'user', 'assistant'

    Returns:
        Formatted prompt string.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    # Add assistant start token for generation
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_chat_prompt(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages into a prompt string for the model.

    Args:
        messages: List of chat message dicts with 'role' and 'content'.
        add_generation_prompt: Whether to append the assistant start token.

    Returns:
        Formatted prompt string.
    """
    prompt = format_chat_messages(messages)
    if not add_generation_prompt and prompt.endswith("<|im_start|>assistant\n"):
        prompt = prompt[: -len("<|im_start|>assistant\n")]
    return prompt