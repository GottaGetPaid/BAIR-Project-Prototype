from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class SimpleTurnGame(ABC):
    """Abstract interface for a simple two-player, turn-based game.

    Implement your game by subclassing and overriding the abstract methods.
    The game should manage its own state and determine legality of actions.
    """

    def __init__(self) -> None:
        self.current_player: int = 0  # 0 or 1

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        ...

    @abstractmethod
    def is_over(self) -> bool:
        """Return True if the game has ended."""
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the state (safe for UI)."""
        ...

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Return available actions for the current player (strings)."""
        ...

    @abstractmethod
    def apply_action(self, action: str) -> Tuple[bool, str]:
        """Apply an action for the current player.

        Returns (ok, message). If not ok, the action is rejected with message.
        Should advance turn on success.
        """
        ...

    @abstractmethod
    def render(self) -> str:
        """Return a concise text summary of the current state."""
        ...
