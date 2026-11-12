from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests

from rag_app.config import Settings


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class OllamaGenerator(BaseGenerator):
    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        response = requests.post(
            url,
            json={
                "model": self.settings.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.settings.generation_temperature,
                },
            },
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("message", {}).get("content", "")).strip()


class TransformersGenerator(BaseGenerator):
    """Optional fully local causal-LM generator using Hugging Face Transformers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_name = self.settings.transformers_generation_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device="cpu",
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.generation_max_new_tokens,
            "do_sample": self.settings.generation_temperature > 0,
            "return_full_text": False,
        }
        if self.settings.generation_temperature > 0:
            generation_kwargs["temperature"] = self.settings.generation_temperature
        output: list[dict[str, Any]] = self._pipeline(prompt, **generation_kwargs)
        return str(output[0]["generated_text"]).strip()


def build_generator(settings: Settings) -> BaseGenerator:
    if settings.llm_provider == "ollama":
        return OllamaGenerator(settings)
    if settings.llm_provider in {"transformers", "hf"}:
        return TransformersGenerator(settings)
    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. Use 'ollama' or 'transformers'."
    )
