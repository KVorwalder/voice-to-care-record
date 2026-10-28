"""Provider-agnostic LLM access via litellm.

The model string decides the backend. The default is a free local model served by Ollama
("ollama_chat/gemma3:12b"); an API model works the same way, e.g. "anthropic/claude-haiku-5-5".
"""

import os
import time
from dataclasses import dataclass
from typing import Protocol

# Default model; can be overridden with the CARE_MODEL environment variable or --model
DEFAULT_MODEL = os.getenv("CARE_MODEL", "ollama_chat/gemma3:12b")


@dataclass
class LLMResponse:
    """What one LLM call returns.

    Fields: text (the raw answer, hopefully JSON), latency_s (duration of the call in seconds).
    """

    text: str
    latency_s: float


class LLM(Protocol):
    """Interface for anything that turns chat messages into text.

    A Protocol only describes the required shape (a `model` name and a `complete` method).
    The real implementation is LiteLLM below; the tests pass a fake that returns fixed answers,
    so they run in milliseconds without a model.
    """

    model: str

    def complete(self, messages: list[dict]) -> LLMResponse: ...


class LiteLLM:
    """Real LLM backend: sends the messages to a model through the litellm library.

    Input (constructor): model string, temperature (0 = as deterministic as possible),
    max_tokens (upper limit for the answer length).
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 4000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict]) -> LLMResponse:
        """Send a chat to the model and return its answer.

        Input:  messages - list of {"role": "system"/"user"/"assistant", "content": text}
        Output: LLMResponse with the answer text and the call duration
        """
        import litellm  # imported lazily: slow import, not needed for tests

        # Arguments for litellm.completion(); the same call works for every provider
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

        # Call the model and measure how long it takes
        start = time.perf_counter()
        resp = litellm.completion(**kwargs)
        latency = time.perf_counter() - start

        # The answer text sits in the first (and only) choice; use "" if the model sent nothing
        return LLMResponse(text=resp.choices[0].message.content or "", latency_s=latency)
