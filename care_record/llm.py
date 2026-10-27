"""Provider-agnostic LLM access via litellm.

One interface for API and local models: the model string decides the backend,
e.g. "anthropic/claude-haiku-5-5", "openai/gpt-4o-mini" or "ollama_chat/qwen2.5:7b".
"""

import os
import time
from dataclasses import dataclass
from typing import Protocol

DEFAULT_API_MODEL = os.getenv("CARE_API_MODEL", "anthropic/claude-haiku-5-5")
DEFAULT_LOCAL_MODEL = os.getenv("CARE_LOCAL_MODEL", "ollama_chat/qwen2.5:7b")


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    """Anything that turns chat messages into text. Tests pass a fake implementation."""

    model: str

    def complete(self, messages: list[dict]) -> LLMResponse: ...


def is_local(model: str) -> bool:
    return model.startswith(("ollama/", "ollama_chat/"))


class LiteLLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4000):
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
        if is_local(self.model):
            # Ollama's JSON mode constrains decoding to valid JSON (not to our schema).
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        resp = litellm.completion(**kwargs)
        latency = time.perf_counter() - start

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            latency_s=latency,
            cost_usd=self._cost(resp, litellm),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )

    def _cost(self, resp, litellm) -> float | None:
        if is_local(self.model):
            return 0.0
        try:
            return float(litellm.completion_cost(completion_response=resp))
        except Exception:
            return None  # model not in litellm's price table
