"""Player data model definitions.

Models describe the shape of important application data. A Player model will
eventually give us one consistent way to represent player identity and team
metadata across services, pages, and analytics code.
"""

from dataclasses import dataclass


@dataclass
class Player:
    """Placeholder model for a baseball player."""

    name: str
    team: str
