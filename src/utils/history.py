from collections import deque
from typing import List, Optional, Literal
from core import Message
from itertools import islice

class GameHistoryDeque:
    """
    A deque of game events (i.e. observations or actions) that can be converted to a list of Message objects.
    """

    def __init__(self):
        self._events = deque()

    def add_action(self, message: str, action: str):
        """Add an action to the history."""
        self._events.append(
            {
                "type": "action",
                "message": message,
                "action": action,
            }
        )

    def add_observation(self, observation: str):
        """Add an observation to the history."""
        self._events.append(
            {
                "type": "observation",
                "observation": observation,
            }
        )

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def _type_to_role(self, event_type: str) -> str:
        """Convert an event type to a role."""
        if event_type == "observation":
            return "user"
        elif event_type == "action":
            return "assistant"
        
    def _event_to_message(self, event: dict, verbose_actions: bool) -> Message:
        """Convert an event to a Message object."""
        if event["type"] == "observation":
            return Message(role="user", content=event["observation"])
        elif event["type"] == "action":
            if verbose_actions:
                return Message(role="assistant", content=event["message"])
            else:
                return Message(role="assistant", content=event["action"])

    def _get_latest_event(self, event_type: Optional[str] = None) -> dict:
        """Get the latest event of a given type."""
        for event in reversed(self._events):
            if event_type is None or event["type"] == event_type:
                return event
        return None

    def get_messages(self, max_history: Optional[int] = None, verbose_actions: Optional[bool] = False) -> List[Message]:
        """Get the list of Message objects representing the history."""

        return [
            self._event_to_message(event, verbose_actions) 
            for event in list(self._events)[-2*max_history:]
        ]

class ChatHistoryDeque:
    """
    A deque of chat events (i.e. messages) that can be converted to a list of Message objects.
    """

    def __init__(self):
        self._events = deque()

    def add_message(self, message: str, role: Literal["user", "assistant"]):
        """Add a message to the history."""
        self._events.append(
            {
                "role": role,
                "message": message,
            }
        )

    def reset(self):
        """Clear the event history."""
        self._events.clear()

    def _event_to_message(self, event: dict) -> Message:
        """Convert an event to a Message object."""
        return Message(role=event["role"], content=event["message"])

    def get_messages(self, max_history: Optional[int] = None) -> List[Message]:
        """Get the list of Message objects representing the history."""

        return [
            self._event_to_message(event) 
            for event in list(self._events)[-2*max_history:]
        ]