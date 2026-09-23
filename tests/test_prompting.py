from types import SimpleNamespace

from LCD.shared.prompt import PROMPT_TEXT

from LCD.shared.lora import _tokenize_training_rows
from LCD.shared.prompting import (
    build_generation_prompt,
    build_ministral_training_texts,
    build_training_texts,
    parse_generated_label,
)


class PlainTokenizer:
    chat_template = None


class MinistralStyleTokenizer:
    chat_template = "ministral-style-test-template"
    eos_token = "</s>"

    def __init__(self) -> None:
        self.rendered_messages = []

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ) -> str:
        assert tokenize is False
        self.rendered_messages.append(messages)
        system = messages[0]["content"]
        conversation = messages[1:]
        rendered = "<s>"
        for index, message in enumerate(conversation):
            if message["role"] == "user":
                content = message["content"]
                if index == len(conversation) - 1:
                    content = f"{system}\n\n{content}"
                rendered += f"[INST]{content}[/INST]"
            else:
                rendered += f"{message['content']}</s>"
        return rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


class OneRowDataframe:
    def __len__(self) -> int:
        return 1

    def itertuples(self, *, index):
        assert index is False
        return iter(
            [
                SimpleNamespace(
                    premise="premise",
                    hypothesis="hypothesis",
                    source="КоАП Статья 1",
                    tag="entailment",
                )
            ]
        )


def test_prompts_use_canonical_constant_once() -> None:
    tokenizer = PlainTokenizer()
    prompt = build_generation_prompt(tokenizer, "premise", "hypothesis", "ternary")
    assert prompt.count(PROMPT_TEXT) == 1


def test_training_text_contains_only_one_system_prompt() -> None:
    prompt, full = build_training_texts(
        PlainTokenizer(), "premise", "hypothesis", "entailment", "ternary"
    )
    assert prompt.count(PROMPT_TEXT) == 1
    assert full.count(PROMPT_TEXT) == 1
    assert full.endswith("Ответ: entailment")


def test_ministral_training_duplicates_prompt_and_preserves_generation_prefix() -> None:
    tokenizer = MinistralStyleTokenizer()

    prompt, full = build_ministral_training_texts(
        tokenizer, "premise", "hypothesis", "entailment", "ternary"
    )

    training_messages = tokenizer.rendered_messages[-1]
    assert training_messages[0]["content"] == PROMPT_TEXT
    assert training_messages[1]["content"].startswith(f"{PROMPT_TEXT}\n\n")
    assert full == f"{prompt}entailment</s>"


def test_ministral_lora_supervises_only_label_and_eos_tokens() -> None:
    rows = _tokenize_training_rows(
        OneRowDataframe(),
        MinistralStyleTokenizer(),
        "ternary",
        4096,
        model_alias="ministral",
    )

    supervised_ids = [token_id for token_id in rows[0]["labels"] if token_id != -100]
    assert "".join(map(chr, supervised_ids)) == "entailment</s>"


def test_parse_generated_labels_is_strict_and_removes_thinking() -> None:
    assert (
        parse_generated_label(
            "<think>maybe entailment</think> not mentioned", "ternary"
        )
        == "not mentioned"
    )
    assert parse_generated_label("not", "ternary") is None
    assert parse_generated_label("no", "ternary") is None
    assert parse_generated_label("no contradiction", "ternary") == "contradiction"
