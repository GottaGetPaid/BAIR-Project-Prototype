from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from config import AppConfig

Message = Dict[str, str]  # {"role": "system|user|assistant", "content": str}


class ModelProvider(ABC):
    """Abstract interface for LLM backends.

    Implementations should accept chat messages and return a string response.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        ...


def load_model_provider() -> ModelProvider:
    """Factory that loads a provider based on env/config."""
    backend = AppConfig.MODEL_BACKEND
    if backend == "huggingface":
        from .huggingface_provider import HuggingFaceProvider
        return HuggingFaceProvider(
            model_id=AppConfig.HF_MODEL_ID,
            device=AppConfig.HF_DEVICE,
            max_new_tokens=AppConfig.HF_MAX_NEW_TOKENS,
            temperature=AppConfig.HF_TEMPERATURE,
        )
    elif backend == "noop":
        from .noop_provider import NoopProvider
        return NoopProvider()
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {backend}")
