# wumpus_engine/core/results.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wumpus_engine.config.schema import SandboxLevel
from wumpus_engine.core.rules import ScoreEvent
from wumpus_sdk.actions import Action
from wumpus_sdk.dto import Outcome
from wumpus_sdk.percept import Percept


class GameStatus(StrEnum):
    '''Engine lifecycle.'''

    READY = 'READY'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'


class TerminationReason(StrEnum):
    '''Why a game ended (spec 4.2.2).'''

    CLIMBED_WITH_GOLD = 'CLIMBED_WITH_GOLD'
    CLIMBED_WITHOUT_GOLD = 'CLIMBED_WITHOUT_GOLD'
    DIED_PIT = 'DIED_PIT'
    DIED_WUMPUS = 'DIED_WUMPUS'
    ACTION_LIMIT = 'ACTION_LIMIT'
    AGENT_TIMEOUT = 'AGENT_TIMEOUT'
    AGENT_CRASH = 'AGENT_CRASH'
    AGENT_ERROR = 'AGENT_ERROR'
    PROTOCOL_VIOLATION = 'PROTOCOL_VIOLATION'
    INVALID_ACTION_LIMIT = 'INVALID_ACTION_LIMIT'
    ENGINE_ERROR = 'ENGINE_ERROR'

    @property
    def outcome(self) -> Outcome:
        '''Post: the coarse value sent to the agent. CLIMBED_WITH_GOLD gives SUCCESS, CLIMBED_WITHOUT_GOLD
        gives CLIMBED_WITHOUT_GOLD, DIED_PIT and DIED_WUMPUS give DIED, ENGINE_ERROR gives ABORTED, every
        other reason gives FORFEIT. The fine-grained cause never reaches the agent.
        '''
        if self is TerminationReason.CLIMBED_WITH_GOLD:
            return Outcome.SUCCESS
        if self is TerminationReason.CLIMBED_WITHOUT_GOLD:
            return Outcome.CLIMBED_WITHOUT_GOLD
        if self in (TerminationReason.DIED_PIT, TerminationReason.DIED_WUMPUS):
            return Outcome.DIED
        if self is TerminationReason.ENGINE_ERROR:
            return Outcome.ABORTED
        return Outcome.FORFEIT


class StepEvent(StrEnum):
    '''What happened during one step; every member is derivable by the agent from its own observations.'''

    TURNED = 'TURNED'
    MOVED = 'MOVED'
    BUMPED = 'BUMPED'
    ARROW_FIRED = 'ARROW_FIRED'
    ARROW_MISSED = 'ARROW_MISSED'
    WUMPUS_KILLED = 'WUMPUS_KILLED'
    NO_ARROWS = 'NO_ARROWS'
    GOLD_GRABBED = 'GOLD_GRABBED'
    NOTHING_TO_GRAB = 'NOTHING_TO_GRAB'
    CLIMBED = 'CLIMBED'
    CLIMB_REJECTED = 'CLIMB_REJECTED'
    DIED = 'DIED'  # the cause (pit or Wumpus) is in HiddenDetail only


# A NamedTuple field named 'count' shadows tuple.count and fails mypy --strict, so ScoreBucket is a
# frozen dataclass with the same (count, total) constructor.
@dataclass(frozen=True, slots=True)
class ScoreBucket:
    '''Number of events of one type and their summed delta.'''

    count: int
    total: int


@dataclass(frozen=True, slots=True)
class ScoreEntry:
    '''One ledger line.'''

    step_index: int
    event: ScoreEvent
    delta: int
    running_total: int


@dataclass(frozen=True, slots=True)
class FinalScore:
    '''Result of ScoreManager.finalize().'''

    total: int
    breakdown: dict[ScoreEvent, ScoreBucket]
    ledger_size: int


@dataclass(frozen=True, slots=True)
class RunInfo:
    '''Everything needed to replay a game exactly. Instructor data: it holds the world seed.'''

    run_id: str  # 32 hex characters, random
    agent_id: str
    scenario_id: str
    world_seed: int  # secret for GRADED scenarios
    agent_seed: int  # drawn from OS entropy; never derived from world_seed
    isolation: SandboxLevel


@dataclass(frozen=True, slots=True)
class StepResult:
    '''Outcome of one applied action.'''

    turn: int  # index of the observation answered
    action: Action
    events: tuple[StepEvent, ...]
    percept: Percept | None  # None if the step was terminal
    score_delta: int
    score_total: int
    terminal: bool
    termination: TerminationReason | None  # set exactly when terminal is True


@dataclass(frozen=True, slots=True)
class GatewayReport:
    '''Teardown report of one agent connection.'''

    exit_code: int | None
    killed_by_engine: bool
    stderr_tail: str  # at most 4096 characters, sanitized
    frames_sent: int
    frames_received: int
    bytes_sent: int
    bytes_received: int
    isolation: SandboxLevel


@dataclass(frozen=True, slots=True)
class GameResult:
    '''Final record of one game.'''

    run: RunInfo
    termination: TerminationReason
    final_score: int  # equals FinalScore.total
    score_breakdown: dict[ScoreEvent, ScoreBucket]
    actions_taken: int
    arrows_used: int
    wumpuses_killed: int
    gold_grabbed: bool
    wall_time_s: float
    agent_name: str | None  # sanitized HelloDTO.agent_name
    fault_message: str | None  # sanitized; INSTRUCTOR audience only
    gateway: GatewayReport | None

    @property
    def won(self) -> bool:
        '''Post: True iff termination is CLIMBED_WITH_GOLD.'''
        return self.termination is TerminationReason.CLIMBED_WITH_GOLD

    @property
    def valid(self) -> bool:
        '''Post: False iff termination is ENGINE_ERROR.'''
        return self.termination is not TerminationReason.ENGINE_ERROR

    @property
    def outcome(self) -> Outcome:
        '''Post: termination.outcome.'''
        return self.termination.outcome