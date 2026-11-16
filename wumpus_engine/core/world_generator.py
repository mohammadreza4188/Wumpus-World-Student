# wumpus_engine/core/world_generator.py
from __future__ import annotations

from typing import ClassVar, Final

from wumpus_engine.config.schema import ScenarioConfig
from wumpus_engine.core.errors import InvalidWorldError, WorldGenerationError
from wumpus_engine.core.world import WorldSpec
from wumpus_sdk.types import (
    GRID_HEIGHT,
    GRID_WIDTH,
    MAX_WUMPUSES,
    MIN_COORD,
    MIN_WUMPUSES,
    START_POSITION,
    Position,
)

_MASK64: Final[int] = (1 << 64) - 1
_GAMMA: Final[int] = 0x9E3779B97F4A7C15
_MIX1: Final[int] = 0xBF58476D1CE4E5B9
_MIX2: Final[int] = 0x94D049BB133111EB

# The 399 cells other than (1, 1), in row-major order: rows (y) outer, columns (x) inner.
_CELLS: Final[tuple[Position, ...]] = tuple(
    Position(x, y)
    for y in range(MIN_COORD, GRID_HEIGHT + 1)
    for x in range(MIN_COORD, GRID_WIDTH + 1)
    if Position(x, y) != START_POSITION
)


class _SplitMix64:
    '''Reference SplitMix64 (Steele, Lea, Flood 2014): self-contained, platform independent.'''

    __slots__ = ('_state',)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK64

    def next_u64(self) -> int:
        '''Post: the next 64-bit output.'''
        self._state = (self._state + _GAMMA) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        '''Pre: bound >= 1. Post: unbiased integer in [0, bound) by rejection sampling.'''
        limit = ((1 << 64) // bound) * bound
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def _shuffled_cells(stream: _SplitMix64) -> list[Position]:
    '''Unbiased Fisher-Yates shuffle of the 399 candidate cells driven by `stream`.'''
    cells = list(_CELLS)
    for i in range(len(cells) - 1, 0, -1):
        j = stream.below(i + 1)
        cells[i], cells[j] = cells[j], cells[i]
    return cells


class WorldGenerator:
    '''Seeded, deterministic world generator.'''

    GENERATOR_VERSION: ClassVar[str] = '1'

    def __init__(self, *, max_attempts: int = 1000) -> None:
        '''Pre: max_attempts >= 1. Raises: ValueError.'''
        if max_attempts < 1:
            raise ValueError('max_attempts must be >= 1')
        self._max_attempts = max_attempts

    def generate(self, scenario: ScenarioConfig, seed: int) -> WorldSpec:
        '''Pre: scenario is validated; 0 <= seed < 2**63.
        Post: an explicit layout is returned as the spec (seed ignored). Otherwise the spec has exactly
        pit_count pits, wumpus_count Wumpuses and one gold on distinct cells other than (1, 1) and passes
        validate(scenario.solvability). Same (scenario, seed, GENERATOR_VERSION) gives the same spec.
        Raises: InvalidWorldError: an explicit layout violates I1 to I5.
        WorldGenerationError: no valid world within max_attempts attempts.
        ValueError: seed out of range.
        '''
        layout = scenario.layout
        if layout is not None:
            pits = frozenset(layout.pits)
            wumpuses = frozenset(layout.wumpuses)
            if len(pits) != len(layout.pits) or len(wumpuses) != len(layout.wumpuses):
                raise InvalidWorldError('I3: an explicit layout lists the same cell twice')
            explicit = WorldSpec(pits=pits, wumpuses=wumpuses, gold=layout.gold)
            explicit.validate(scenario.solvability)
            return explicit
        if not 0 <= seed < 1 << 63:
            raise ValueError('seed must satisfy 0 <= seed < 2**63')
        wumpus_count, pit_count = scenario.wumpus_count, scenario.pit_count
        if not MIN_WUMPUSES <= wumpus_count <= MAX_WUMPUSES:
            raise WorldGenerationError(f'wumpus_count {wumpus_count} is outside 1..4')
        if pit_count < 0 or 1 + wumpus_count + pit_count > len(_CELLS):
            raise WorldGenerationError(f'pit_count {pit_count} does not fit next to the other items')
        seeder = _SplitMix64(seed)
        for _attempt in range(self._max_attempts):
            cells = _shuffled_cells(_SplitMix64(seeder.next_u64()))
            spec = WorldSpec(
                pits=frozenset(cells[1 + wumpus_count : 1 + wumpus_count + pit_count]),
                wumpuses=frozenset(cells[1 : 1 + wumpus_count]),
                gold=cells[0],
            )
            try:
                spec.validate(scenario.solvability)
            except InvalidWorldError:
                continue
            return spec
        raise WorldGenerationError(
            f'no {scenario.solvability.value} world with {pit_count} pits and {wumpus_count} '
            f'Wumpuses in {self._max_attempts} attempts'
        )