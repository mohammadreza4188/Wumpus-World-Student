# wumpus_sdk/agent.py
from __future__ import annotations

import random
from abc import ABC, abstractmethod

from wumpus_sdk.actions import Action
from wumpus_sdk.dto import (
    GameOverDTO,
    InvalidActionDTO,
    ObservationDTO,
    SessionInitDTO,
)


class Agent(ABC):
    """Base class for all Wumpus World agents."""

    rng: random.Random

    def __init__(self) -> None:
        self.rng = random.Random()

    def _bind_session(self, session: SessionInitDTO) -> None:
        self.rng = random.Random(session.agent_seed)

    def on_game_start(self, session: SessionInitDTO) -> None:
        return None

    def on_invalid_action(self, notice: InvalidActionDTO) -> None:
        return None

    def on_game_over(self, result: GameOverDTO) -> None:
        return None

    @abstractmethod
    def act(self, observation: ObservationDTO) -> Action:
        ...