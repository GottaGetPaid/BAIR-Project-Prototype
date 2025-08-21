from __future__ import annotations
from typing import Any, Dict, List, Tuple

from .base_game import SimpleTurnGame


class MyTemplateGame(SimpleTurnGame):
    """A minimal skeleton to copy and modify for your custom game.

    Rules (example you can replace):
    - Players take turns to append either "A" or "B" to a shared string.
    - First to make the string length reach 6 ends the game.
    This is only a reference; feel free to discard and implement your own logic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.s: str = ""

    def reset(self) -> None:
        self.s = ""
        self.current_player = 0

    def is_over(self) -> bool:
        return len(self.s) >= 6

    def get_state(self) -> Dict[str, Any]:
        return {
            "sequence": self.s,
            "current_player": self.current_player,
            "length": len(self.s),
        }

    def get_available_actions(self) -> List[str]:
        return ["A", "B"]

    def apply_action(self, action: str) -> Tuple[bool, str]:
        action = action.strip().upper()
        if action not in ("A", "B"):
            return False, "Invalid action. Choose A or B."
        self.s += action
        self.current_player = 1 - self.current_player
        return True, "OK"

    def render(self) -> str:
        return f"Seq: {self.s} (len={len(self.s)}), to-move=P{self.current_player}"
