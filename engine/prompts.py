from __future__ import annotations
from typing import List, Dict

Message = Dict[str, str]


def build_turn_messages(state_text: str, actions: List[str]) -> List[Message]:
    instructions = (
        "You are playing a simple turn-based game.\n"
        "Only respond with your chosen action, exactly matching one of the allowed actions.\n"
        "Do not add any extra words."
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
