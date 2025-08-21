from __future__ import annotations
from typing import List, Dict, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception as e:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

Message = Dict[str, str]


def _format_chat(messages: List[Message], system_prompt: Optional[str]) -> str:
    """Fallback chat to text if tokenizer.chat_template is unavailable."""
    lines = []
    if system_prompt:
        lines.append(f"<system>\n{system_prompt}\n</system>")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"<{role}>\n{content}\n</{role}>")
    lines.append("<assistant>\n")
    return "\n".join(lines)


class HuggingFaceProvider:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        if pipeline is None:
            raise RuntimeError(
                "transformers is not available. Install dependencies from requirements.txt"
            )
        if not model_id:
            raise ValueError("HF_MODEL_ID must be set for huggingface backend")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        tok = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = tok

        text_gen = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tok,
            device_map=device,
            torch_dtype=None,
        )
        self.pipe = text_gen

    def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            prompt_text = tok.apply_chat_template(
                messages if system_prompt is None else ([{"role": "system", "content": system_prompt}] + messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = _format_chat(messages, system_prompt)

        out = self.pipe(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # transformers pipelines return list[dict]
        text = out[0]["generated_text"]
        # Strip the prompt if present (basic heuristic)
        if text.startswith(prompt_text):
            text = text[len(prompt_text):]
        return text.strip()
