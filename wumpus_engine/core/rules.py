# wumpus_engine/core/rules.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

__all__ = ['DEFAULT_SCORE_TABLE', 'ScoreEvent', 'ScoreTable', 'SolvabilityPolicy']


class ScoreEvent(StrEnum):
    '''Events that carry a price in the score table.'''

    ACTION = 'ACTION'  # every action taken, including no-ops and the fatal one
    SHOOT_ARROW = 'SHOOT_ARROW'  # extra cost, only when an arrow was actually fired
    GOLD_GRAB = 'GOLD_GRAB'
    WUMPUS_KILL = 'WUMPUS_KILL'
    DEATH = 'DEATH'
    FORFEIT = 'FORFEIT'  # non-voluntary end: timeout, crash, protocol violation, action limit


class SolvabilityPolicy(StrEnum):
    '''How strictly WorldSpec.validate() checks that the gold can be reached.'''

    NONE = 'NONE'  # no check
    REACHABLE_WITH_SHOOTING = 'REACHABLE_WITH_SHOOTING'  # reachable over non-pit cells
    SAFE_PATH = 'SAFE_PATH'  # reachable over cells with neither pit nor Wumpus


@dataclass(frozen=True, slots=True)
class ScoreTable:
    '''Signed deltas. Defaults are the specification: a SHOOT totals -11 (action -1 plus shoot_extra -10).'''

    action: int = -1
    shoot_extra: int = -10
    gold_grab: int = 1000
    wumpus_kill: int = 0
    death: int = -1000
    forfeit: int = -1000

    def __post_init__(self) -> None:
        '''Raises: ValueError: unless every value is an int, penalties are <= 0 and rewards are >= 0.'''
        penalties = (
            ('action', self.action),
            ('shoot_extra', self.shoot_extra),
            ('death', self.death),
            ('forfeit', self.forfeit),
        )
        rewards = (('gold_grab', self.gold_grab), ('wumpus_kill', self.wumpus_kill))
        for name, value in (*penalties, *rewards):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an int, got {value!r}')
        for name, value in penalties:
            if value > 0:
                raise ValueError(f'{name} is a penalty and must be <= 0, got {value}')
        for name, value in rewards:
            if value < 0:
                raise ValueError(f'{name} is a reward and must be >= 0, got {value}')

    def delta_for(self, event: ScoreEvent) -> int:
        '''Pre: none. Post: the signed delta of `event`. Raises: nothing.'''
        match event:
            case ScoreEvent.ACTION:
                return self.action
            case ScoreEvent.SHOOT_ARROW:
                return self.shoot_extra
            case ScoreEvent.GOLD_GRAB:
                return self.gold_grab
            case ScoreEvent.WUMPUS_KILL:
                return self.wumpus_kill
            case ScoreEvent.DEATH:
                return self.death
            case ScoreEvent.FORFEIT:
                return self.forfeit
            case _:
                assert_never(event)


DEFAULT_SCORE_TABLE: Final[ScoreTable] = ScoreTable()