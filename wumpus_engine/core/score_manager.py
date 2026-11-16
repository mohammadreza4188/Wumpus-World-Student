# wumpus_engine/core/score_manager.py
from __future__ import annotations

from wumpus_engine.core.errors import ScoreFinalizedError
# results.ScoreBucket must be a frozen dataclass(count, total): a NamedTuple field named 'count'
# shadows tuple.count and fails mypy --strict.
from wumpus_engine.core.results import FinalScore, ScoreBucket, ScoreEntry
from wumpus_engine.core.rules import DEFAULT_SCORE_TABLE, ScoreEvent, ScoreTable

__all__ = ['ScoreEvent', 'ScoreManager', 'ScoreTable']


class ScoreManager:
    '''Append-only score ledger for one game.'''

    def __init__(self, table: ScoreTable = DEFAULT_SCORE_TABLE) -> None:
        '''Pre: none. Post: empty ledger, total 0, not finalized. Raises: nothing.'''
        self._table = table
        self._entries: list[ScoreEntry] = []
        self._total = 0
        self._finalized = False

    @property
    def table(self) -> ScoreTable:
        '''Post: the immutable price list in force.'''
        return self._table

    @property
    def total(self) -> int:
        '''Post: sum of all deltas so far; always equals sum(e.delta for e in entries).'''
        return self._total

    @property
    def entries(self) -> tuple[ScoreEntry, ...]:
        '''Post: the ledger, oldest first.'''
        return tuple(self._entries)

    @property
    def is_finalized(self) -> bool:
        '''Post: True after finalize().'''
        return self._finalized

    def record(self, event: ScoreEvent, *, step_index: int) -> ScoreEntry:
        '''Pre: not finalized; step_index is not smaller than the previous entry's step_index.
        Post: total changed by table.delta_for(event); the entry is stored and returned.
        Raises: ScoreFinalizedError: after finalize(). ValueError: step_index decreased.
        '''
        if self._finalized:
            raise ScoreFinalizedError('the score ledger is finalized')
        if self._entries and step_index < self._entries[-1].step_index:
            raise ValueError('step_index must not decrease')
        delta = self._table.delta_for(event)
        self._total += delta
        entry = ScoreEntry(
            step_index=step_index, event=event, delta=delta, running_total=self._total
        )
        self._entries.append(entry)
        return entry

    def delta_for_step(self, step_index: int) -> int:
        '''Pre: none. Post: sum of the deltas recorded for `step_index`; 0 if none. Raises: nothing.'''
        return sum(entry.delta for entry in self._entries if entry.step_index == step_index)

    def breakdown(self) -> dict[ScoreEvent, ScoreBucket]:
        '''Pre: none. Post: count and sum per event type that occurred; others omitted. Raises: nothing.'''
        counts: dict[ScoreEvent, int] = {}
        totals: dict[ScoreEvent, int] = {}
        for entry in self._entries:
            counts[entry.event] = counts.get(entry.event, 0) + 1
            totals[entry.event] = totals.get(entry.event, 0) + entry.delta
        return {
            event: ScoreBucket(count=counts[event], total=totals[event])
            for event in ScoreEvent
            if event in counts
        }

    def finalize(self) -> FinalScore:
        '''Pre: none; idempotent. Post: record() now raises; FinalScore.total == self.total.
        Raises: nothing.
        '''
        self._finalized = True
        return FinalScore(
            total=self._total, breakdown=self.breakdown(), ledger_size=len(self._entries)
        )