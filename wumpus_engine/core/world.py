# wumpus_engine/core/world.py
from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, assert_never

from wumpus_engine.core.errors import InvalidWorldError
from wumpus_engine.core.rules import SolvabilityPolicy
from wumpus_sdk.errors import ProtocolError
from wumpus_sdk.percept import Percept
from wumpus_sdk.types import MAX_WUMPUSES, MIN_WUMPUSES, START_POSITION, Direction, Position


def _gold_reachable(gold: Position, blocked: frozenset[Position]) -> bool:
    '''Breadth-first search from (1, 1) over 4-neighbour moves that avoid `blocked`.'''
    if START_POSITION in blocked:
        return False
    seen = {START_POSITION}
    queue = deque([START_POSITION])
    while queue:
        current = queue.popleft()
        if current == gold:
            return True
        for cell in current.neighbors():
            if cell not in seen and cell not in blocked:
                seen.add(cell)
                queue.append(cell)
    return False


def _parse_cells(raw: object) -> frozenset[Position]:
    '''Raises: ProtocolError: unless raw is a list of distinct valid [x, y] pairs.'''
    if not isinstance(raw, list):
        raise ProtocolError('expected a list of cells')
    cells = [Position.from_wire(item) for item in raw]
    if len(set(cells)) != len(cells):
        raise ProtocolError('duplicate cell in list')
    return frozenset(cells)


@dataclass(frozen=True, slots=True)
class WorldSpec:
    '''Immutable hidden layout. Produced by WorldGenerator or built from an ExplicitLayout.'''

    pits: frozenset[Position]
    wumpuses: frozenset[Position]
    gold: Position

    @property
    def arrows(self) -> int:
        '''Post: initial arrow count, equal to len(self.wumpuses) (spec section 2).'''
        return len(self.wumpuses)

    def validate(
        self, policy: SolvabilityPolicy = SolvabilityPolicy.REACHABLE_WITH_SHOOTING
    ) -> None:
        '''Pre: none. Post: returns only if invariants I1 to I5 hold; the spec is not modified.
        Raises: InvalidWorldError: the message names the first violated invariant.
        '''
        if START_POSITION in self.pits:
            raise InvalidWorldError('I1: the start cell (1, 1) holds a pit')
        if START_POSITION in self.wumpuses:
            raise InvalidWorldError('I1: the start cell (1, 1) holds a Wumpus')
        if self.gold == START_POSITION:
            raise InvalidWorldError('I1: the start cell (1, 1) holds the gold')
        if not MIN_WUMPUSES <= len(self.wumpuses) <= MAX_WUMPUSES:
            raise InvalidWorldError(
                f'I2: Wumpus count {len(self.wumpuses)} is outside {MIN_WUMPUSES}..{MAX_WUMPUSES}'
            )
        if self.pits & self.wumpuses:
            raise InvalidWorldError('I3: a pit and a Wumpus share a cell')
        if self.gold in self.pits:
            raise InvalidWorldError('I3: the gold shares a cell with a pit')
        if self.gold in self.wumpuses:
            raise InvalidWorldError('I3: the gold shares a cell with a Wumpus')
        blocked: frozenset[Position]
        match policy:
            case SolvabilityPolicy.NONE:
                return
            case SolvabilityPolicy.REACHABLE_WITH_SHOOTING:
                blocked = self.pits
            case SolvabilityPolicy.SAFE_PATH:
                blocked = self.pits | self.wumpuses
            case _:
                assert_never(policy)
        if not _gold_reachable(self.gold, blocked):
            raise InvalidWorldError(f'I5: the gold is not reachable under policy {policy.value}')

    def to_json(self) -> dict[str, object]:
        '''Pre: none. Post: {'pits': [[x, y], ...], 'wumpuses': [...], 'gold': [x, y]}, every list sorted.
        Raises: nothing.
        '''
        return {
            'pits': [cell.to_wire() for cell in sorted(self.pits)],
            'wumpuses': [cell.to_wire() for cell in sorted(self.wumpuses)],
            'gold': self.gold.to_wire(),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> WorldSpec:
        '''Pre: none. Post: a spec equal to the one that produced `raw`.
        Raises: InvalidWorldError: on any shape, type or range error.
        '''
        if not isinstance(raw, Mapping) or set(raw) != {'pits', 'wumpuses', 'gold'}:
            raise InvalidWorldError('world JSON needs exactly the keys pits, wumpuses and gold')
        try:
            return cls(
                pits=_parse_cells(raw['pits']),
                wumpuses=_parse_cells(raw['wumpuses']),
                gold=Position.from_wire(raw['gold']),
            )
        except ProtocolError as exc:
            raise InvalidWorldError(f'invalid world JSON: {exc}') from exc


@dataclass(frozen=True, slots=True)
class ShotResult:
    '''Outcome of one arrow.'''

    path: tuple[Position, ...]  # cells the arrow entered, nearest first
    killed_at: Position | None  # cell of the Wumpus that died, None on a miss

    @property
    def hit(self) -> bool:
        '''Post: True iff killed_at is not None.'''
        return self.killed_at is not None


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    '''Immutable copy of the current hidden state, for the Observer only.'''

    pits: frozenset[Position]
    wumpuses_alive: frozenset[Position]
    wumpuses_dead: frozenset[Position]
    gold_position: Position
    gold_taken: bool


class World:
    '''The live hidden world. Exists only in the engine process.'''

    __slots__ = ('_pits', '_wumpuses_initial', '_wumpuses_dead', '_gold', '_gold_taken', '_fingerprint')

    def __init__(self, spec: WorldSpec) -> None:
        '''Pre: spec.validate(SolvabilityPolicy.NONE) passes (re-checked here).
        Post: every Wumpus alive, gold untaken. Raises: InvalidWorldError.
        '''
        spec.validate(SolvabilityPolicy.NONE)
        self._pits: frozenset[Position] = spec.pits
        self._wumpuses_initial: frozenset[Position] = spec.wumpuses
        self._wumpuses_dead: set[Position] = set()
        self._gold: Position = spec.gold
        self._gold_taken = False
        canonical = json.dumps(spec.to_json(), sort_keys=True, separators=(',', ':'))
        self._fingerprint = hashlib.sha256(canonical.encode('ascii')).hexdigest()

    @property
    def wumpuses_alive(self) -> int:
        '''Post: number of living Wumpuses.'''
        return len(self._wumpuses_initial) - len(self._wumpuses_dead)

    @property
    def gold_taken(self) -> bool:
        '''Post: True once take_gold() has returned True.'''
        return self._gold_taken

    def is_pit(self, position: Position) -> bool:
        '''Pre: none. Post: pure. Raises: nothing.'''
        return position in self._pits

    def is_living_wumpus(self, position: Position) -> bool:
        '''Pre: none. Post: True iff a Wumpus started on `position` and has not been killed.'''
        return position in self._wumpuses_initial and position not in self._wumpuses_dead

    def has_gold_at(self, position: Position) -> bool:
        '''Pre: none. Post: True iff the gold is on `position` and has not been taken.'''
        return not self._gold_taken and position == self._gold

    def perceive(
        self, position: Position, *, bump: bool = False, scream: bool = False
    ) -> Percept:
        '''Pre: `position` is a cell the agent occupies.
        Post: pure. breeze: a neighbouring pit. stench: a neighbouring LIVING Wumpus.
        glitter: untaken gold here. bump and scream are copied from the arguments.
        '''
        neighbours = position.neighbors()
        return Percept(
            breeze=any(cell in self._pits for cell in neighbours),
            stench=any(self.is_living_wumpus(cell) for cell in neighbours),
            glitter=self.has_gold_at(position),
            scream=scream,
            bump=bump,
        )

    def fire_arrow(self, origin: Position, direction: Direction) -> ShotResult:
        '''Pre: the caller already deducted the arrow and the scores.
        Post: the ray starts next to `origin` and runs to the boundary. The first LIVING Wumpus on it
        dies (at most one per call). Dead Wumpuses and pits do not block. A miss changes nothing.
        '''
        path: list[Position] = []
        cell = origin.neighbor(direction)
        while cell is not None:
            path.append(cell)
            if self.is_living_wumpus(cell):
                self._wumpuses_dead.add(cell)
                return ShotResult(path=tuple(path), killed_at=cell)
            cell = cell.neighbor(direction)
        return ShotResult(path=tuple(path), killed_at=None)

    def take_gold(self, position: Position) -> bool:
        '''Pre: none. Post: True at most once per world; afterwards has_gold_at() is False everywhere.'''
        if not self.has_gold_at(position):
            return False
        self._gold_taken = True
        return True

    def snapshot(self) -> WorldSnapshot:
        '''Pre: none. Post: an immutable copy; later mutations do not affect it.'''
        return WorldSnapshot(
            pits=self._pits,
            wumpuses_alive=frozenset(self._wumpuses_initial - self._wumpuses_dead),
            wumpuses_dead=frozenset(self._wumpuses_dead),
            gold_position=self._gold,
            gold_taken=self._gold_taken,
        )

    def fingerprint(self) -> str:
        '''Pre: none. Post: sha256 hex of the canonical JSON of the INITIAL layout; unchanged by play.'''
        return self._fingerprint

    def __repr__(self) -> str:
        '''Post: always '<World: hidden>'.'''
        return '<World: hidden>'

    def __reduce__(self) -> NoReturn:
        '''Raises: TypeError, always. A World cannot be pickled, copied or deep-copied.'''
        raise TypeError('a World cannot be pickled or copied')