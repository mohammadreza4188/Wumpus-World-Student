# wumpus_sdk/dto.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeAlias

from wumpus_sdk.percept import Percept
from wumpus_sdk.types import Direction, Position

PROTOCOL_VERSION: Final[int] = 1


class Outcome(StrEnum):
    '''Coarse game result sent to the agent; the fine-grained cause never crosses the boundary.'''

    SUCCESS = 'SUCCESS'  # climbed out holding the gold
    CLIMBED_WITHOUT_GOLD = 'CLIMBED_WITHOUT_GOLD'
    DIED = 'DIED'  # pit or Wumpus; the cause is never disclosed
    FORFEIT = 'FORFEIT'  # timeout, crash, protocol violation, action limit, strikes
    ABORTED = 'ABORTED'  # cancelled by the framework; not charged to the agent


# ---- framework to agent -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionInitDTO:
    '''Public rules, sent once after the handshake.'''

    protocol_version: int
    grid_width: int
    grid_height: int
    start_position: Position
    start_direction: Direction
    initial_arrows: int
    max_actions: int
    action_timeout_s: float
    score_rules: dict[str, int]
    agent_seed: int


@dataclass(frozen=True, slots=True)
class ObservationDTO:
    '''What the agent may know now: operational data plus the five percepts.'''

    step_index: int
    position: Position
    direction: Direction
    arrows_remaining: int
    has_gold: bool
    percept: Percept


@dataclass(frozen=True, slots=True)
class InvalidActionDTO:
    '''Replaces the observation when the engine rejects a reply.'''

    step_index: int
    reason: str  # UNKNOWN_ACTION or STEP_MISMATCH
    strikes_used: int
    strikes_allowed: int


@dataclass(frozen=True, slots=True)
class GameOverDTO:
    '''Terminal message, sent best effort.'''

    outcome: Outcome
    final_score: int
    actions_taken: int


# ---- agent to framework -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HelloDTO:
    '''First frame of the child process.'''

    protocol_version: int
    sdk_version: str
    agent_name: str


@dataclass(frozen=True, slots=True)
class ActionDTO:
    '''The agent's choice for one step.'''

    step_index: int
    action: str  # raw string on purpose: the engine, not the codec, judges it with Action.parse


@dataclass(frozen=True, slots=True)
class AgentErrorDTO:
    '''Terminal error report; also sent in place of HelloDTO when the student module fails to load.'''

    step_index: int
    error_type: str
    message: str
    traceback: str


WireMessage: TypeAlias = (
    SessionInitDTO | ObservationDTO | InvalidActionDTO | GameOverDTO
    | HelloDTO | ActionDTO | AgentErrorDTO
)

OUTBOUND_TYPES: Final[frozenset[type]] = frozenset(
    {SessionInitDTO, ObservationDTO, InvalidActionDTO, GameOverDTO}
)
INBOUND_TYPES: Final[frozenset[type]] = frozenset({HelloDTO, ActionDTO, AgentErrorDTO})