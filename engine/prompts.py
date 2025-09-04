from __future__ import annotations
from typing import List, Dict

Message = Dict[str, str]


def build_turn_messages(state_text: str, actions: List[str]) -> List[Message]:
    instructions = (
	"Using the context provided, answer the user's questions.\n"
        "Choose your responses, making sure to provide comprehensive answers.\n"
   	"Do not add any extra words; be as concise as possible."
    )
    user = (
        f"State:\n{state_text}\n\n"
        f"Actions: [{', '.join(actions)}]\n"
        f"Your turn: choose one action."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]
