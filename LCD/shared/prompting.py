"""Prompt construction and strict generated-label parsing."""

from __future__ import annotations

import re
from typing import Any

from .prompt import PROMPT_TEXT

from .common import LABELS_BY_TASK, Task


def prompt_for_task(task: Task) -> str:
    """Return the imported canonical prompt for a task."""
    return PROMPT_TEXT


def build_messages(
    premise: str,
    hypothesis: str,
    task: Task,
    *,
    assistant_label: str | None = None,
    prompt_text: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages without duplicating the system prompt."""
    messages = [
        {"role": "system", "content": prompt_text or prompt_for_task(task)},
        {
            "role": "user",
            "content": f"Предпосылка: {premise}\nГипотеза: {hypothesis}",
        },
    ]
    if assistant_label is not None:
        messages.append({"role": "assistant", "content": assistant_label})
    return messages


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply a tokenizer chat template, with a plain-text fallback."""
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)

    system = messages[0]["content"]
    user = messages[1]["content"]
    if len(messages) == 3:
        return f"{system}\n\n{user}\nОтвет: {messages[2]['content']}"
    return f"{system}\n\n{user}\nОтвет:"


def build_generation_prompt(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    task: Task,
    *,
    prompt_text: str | None = None,
) -> str:
    """Build one inference prompt from the canonical imported task prompt."""
    return apply_chat_template(
        tokenizer,
        build_messages(premise, hypothesis, task, prompt_text=prompt_text),
        add_generation_prompt=True,
    )


def build_training_texts(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    label: str,
    task: Task,
) -> tuple[str, str]:
    """Return prompt-only and prompt-plus-label strings for response-only loss."""
    if label not in LABELS_BY_TASK[task]:
        raise ValueError(f"Invalid {task} label: {label!r}")
    prompt = build_generation_prompt(tokenizer, premise, hypothesis, task)
    full = apply_chat_template(
        tokenizer,
        build_messages(premise, hypothesis, task, assistant_label=label),
        add_generation_prompt=False,
    )
    return prompt, full


def build_ministral_training_texts(
    tokenizer: Any,
    premise: str,
    hypothesis: str,
    label: str,
    task: Task,
) -> tuple[str, str]:
    """Return prefix-compatible Ministral prompt and labeled training text.

    Ministral's chat template omits the system message when an assistant turn
    follows the user turn. Duplicating the canonical prompt at the start of the
    training user message preserves it in the rendered conversation. The two
    newlines match the separator used by the same template for generation, so
    the labeled text is an exact continuation of the inference prompt.
    """
    if label not in LABELS_BY_TASK[task]:
        raise ValueError(f"Invalid {task} label: {label!r}")
    prompt = build_generation_prompt(tokenizer, premise, hypothesis, task)
    messages = build_messages(
        premise,
        hypothesis,
        task,
        assistant_label=label,
    )
    messages[1]["content"] = (
        f"{prompt_for_task(task)}\n\n{messages[1]['content']}"
    )
    full = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError(
            "Ministral training text is not an exact continuation of its "
            "generation prompt; the tokenizer chat template may have changed"
        )
    return prompt, full


def parse_generated_label(text: str, task: Task) -> str | None:
    """Parse exactly one supported label from generated model text."""
    from .common import validate_task

    validate_task(task)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    normalized = cleaned.lower().strip()
    patterns = (
        ("not mentioned", r"\bnot\s+mentioned\b"),
        ("contradiction", r"\bcontradiction\b"),
        ("entailment", r"\bentailment\b"),
    )
    matches = [label for label, pattern in patterns if re.search(pattern, normalized)]
    return matches[0] if len(matches) == 1 else None
