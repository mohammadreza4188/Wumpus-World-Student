# wumpus_sdk/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from wumpus_sdk.errors import ProtocolError

GRID_WIDTH: Final[int] = 20
GRID_HEIGHT: Final[int] = 20
MIN_COORD: Final[int] = 1
MIN_WUMPUSES: Final[int] = 1
MAX_WUMPUSES: Final[int] = 4


class Direction(StrEnum):
    '''Compass directions: NORTH is y + 1, EAST is x + 1, SOUTH is y - 1, WEST is x - 1.'''

    NORTH = 'NORTH'
    EAST = 'EAST'
    SOUTH = 'SOUTH'
    WEST = 'WEST'

    @property
    def delta(self) -> tuple[int, int]:
        '''Post: unit step (dx, dy): NORTH (0, 1), EAST (1, 0), SOUTH (0, -1), WEST (-1, 0).'''
        return _DELTAS[self]

    def turn_left(self) -> Direction:
        '''Post: 90 degrees counter-clockwise: EAST -> NORTH -> WEST -> SOUTH -> EAST.'''
        return _CLOCKWISE[(_CLOCKWISE.index(self) - 1) % len(_CLOCKWISE)]

    def turn_right(self) -> Direction:
        '''Post: 90 degrees clockwise: EAST -> SOUTH -> WEST -> NORTH -> EAST.'''
        return _CLOCKWISE[(_CLOCKWISE.index(self) + 1) % len(_CLOCKWISE)]


_CLOCKWISE: Final[tuple[Direction, ...]] = (
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
)
_DELTAS: Final[dict[Direction, tuple[int, int]]] = {
    Direction.NORTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, -1),
    Direction.WEST: (-1, 0),
}


@dataclass(frozen=True, slots=True, order=True)
class Position:
    '''A cell. x is the column (1..20, grows EAST); y is the row (1..20, grows NORTH).'''

    x: int
    y: int

    def __post_init__(self) -> None:
        '''Raises: ValueError: unless x and y are ints (bool rejected) within 1..20.'''
        for name, value, upper in (('x', self.x, GRID_WIDTH), ('y', self.y, GRID_HEIGHT)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an int, got {value!r}')
            if not MIN_COORD <= value <= upper:
                raise ValueError(f'{name}={value} is outside {MIN_COORD}..{upper}')

    def neighbor(self, direction: Direction) -> Position | None:
        '''Post: the edge-adjacent cell in `direction`, or None if it lies outside the grid.'''
        dx, dy = direction.delta
        nx, ny = self.x + dx, self.y + dy
        if not (MIN_COORD <= nx <= GRID_WIDTH and MIN_COORD <= ny <= GRID_HEIGHT):
            return None
        return Position(nx, ny)

    def neighbors(self) -> tuple[Position, ...]:
        '''Post: in-bounds edge-adjacent cells ordered NORTH, EAST, SOUTH, WEST (2 to 4 items).'''
        cells = (self.neighbor(direction) for direction in _CLOCKWISE)
        return tuple(cell for cell in cells if cell is not None)

    def to_wire(self) -> list[int]:
        '''Post: [x, y].'''
        return [self.x, self.y]

    @classmethod
    def from_wire(cls, raw: object) -> Position:
        '''Raises: ProtocolError: unless raw is a list of exactly two in-range ints.'''
        if not isinstance(raw, list) or len(raw) != 2:
            raise ProtocolError('a position must be a list of exactly two integers')
        try:
            return cls(raw[0], raw[1])
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc


START_POSITION: Final[Position] = Position(1, 1)
START_DIRECTION: Final[Direction] = Direction.EAST