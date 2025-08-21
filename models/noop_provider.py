from __future__ import annotations
from typing import List, Dict, Optional

Message = Dict[str, str]


class NoopProvider:
    """A simple dev backend that returns a fixed or heuristic response."""

    def generate(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # Try to choose first available action if encoded in the last user message
        last = messages[-1]["content"] if messages else ""
        # naive parse for a line like: Actions: [a, b, c]
        marker = "Actions:"
        if marker in last:
            after = last.split(marker, 1)[1].strip()
            if after.startswith("[") and "]" in after:
                choices = after[1:after.index("]")].split(",")
                if choices:
                    return choices[0].strip()
        return "pass"
