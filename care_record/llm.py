"""Provider-agnostic LLM access via litellm.

The model string decides the backend. The default is a free local model served by Ollama
("ollama_chat/gemma3:12b"); an API model works the same way, e.g. "anthropic/claude-haiku-5-5".
"""

import os
import time
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL = os.getenv("CARE_MODEL", "ollama_chat/gemma3:12b")


@dataclass
class LLMResponse:
    text: str
    latency_s: float


class LLM(Protocol):
    """Anything that turns chat messages into text. Tests pass a fake implementation."""

    model: str

    def complete(self, messages: list[dict]) -> LLMResponse: ...


class LiteLLM:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 4000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict]) -> LLMResponse:
        import litellm  # imported lazily: slow import, not needed for tests

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "drop_params": True,  # silently drop params a provider does not support
        }
        if self.model.startswith(("ollama/", "ollama_chat/")):
            # Ollama's JSON mode constrains decoding to valid JSON (not to our schema).
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        resp = litellm.completion(**kwargs)
        latency = time.perf_counter() - start
        return LLMResponse(text=resp.choices[0].message.content or "", latency_s=latency)
