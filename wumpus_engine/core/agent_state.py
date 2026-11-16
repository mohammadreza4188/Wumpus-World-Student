# wumpus_engine/core/agent_state.py
from __future__ import annotations

from dataclasses import dataclass, field

from wumpus_sdk.types import START_DIRECTION, START_POSITION, Direction, Position


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    '''Immutable copy of the agent's body.'''

    position: Position
    direction: Direction
    arrows: int
    has_gold: bool
    alive: bool
    exited: bool
    actions_taken: int
    visited: frozenset[Position]
    path: tuple[Position, ...]  # START_POSITION, then every cell entered after a successful move


@dataclass(slots=True)
class AgentState:
    '''The agent's body. Owned and mutated only by GameEngine.'''

    arrows: int  # initial value: the number of Wumpuses
    position: Position = START_POSITION
    direction: Direction = START_DIRECTION
    has_gold: bool = False
    alive: bool = True
    exited: bool = False
    actions_taken: int = 0
    visited: set[Position] = field(default_factory=lambda: {START_POSITION})
    path: list[Position] = field(default_factory=lambda: [START_POSITION])

    def _require_active(self) -> None:
        '''Raises: ValueError: the agent is dead or has exited.'''
        if not self.alive or self.exited:
            raise ValueError('the agent is no longer active')

    def turn_left(self) -> None:
        '''Pre: alive and not exited. Post: direction rotated 90 degrees counter-clockwise.
        Raises: ValueError: precondition violated.
        '''
        self._require_active()
        self.direction = self.direction.turn_left()

    def turn_right(self) -> None:
        '''Pre: alive and not exited. Post: direction rotated 90 degrees clockwise.
        Raises: ValueError: precondition violated.
        '''
        self._require_active()
        self.direction = self.direction.turn_right()

    def move_to(self, position: Position) -> None:
        '''Pre: alive and not exited. Post: position set; added to visited and appended to path.
        Raises: ValueError: precondition violated.
        '''
        self._require_active()
        self.position = position
        self.visited.add(position)
        self.path.append(position)

    def consume_arrow(self) -> None:
        '''Pre: arrows >= 1. Post: arrows reduced by one. Raises: ValueError: arrows is 0.'''
        if self.arrows < 1:
            raise ValueError('no arrows left')
        self.arrows -= 1

    def pick_up_gold(self) -> None:
        '''Pre: alive. Post: has_gold is True. Raises: ValueError: the agent is dead.'''
        if not self.alive:
            raise ValueError('a dead agent cannot pick up gold')
        self.has_gold = True

    def die(self) -> None:
        '''Pre: none. Post: alive is False. Raises: nothing.'''
        self.alive = False

    def exit_cave(self) -> None:
        '''Pre: alive and position == START_POSITION. Post: exited is True.
        Raises: ValueError: precondition violated.
        '''
        if not self.alive or self.position != START_POSITION:
            raise ValueError('the agent can only exit alive from (1, 1)')
        self.exited = True

    def snapshot(self) -> AgentSnapshot:
        '''Pre: none. Post: an immutable copy; later mutations do not affect it.'''
        return AgentSnapshot(
            position=self.position,
            direction=self.direction,
            arrows=self.arrows,
            has_gold=self.has_gold,
            alive=self.alive,
            exited=self.exited,
            actions_taken=self.actions_taken,
            visited=frozenset(self.visited),
            path=tuple(self.path),
        )