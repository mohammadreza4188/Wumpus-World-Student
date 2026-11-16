# wumpus_engine/core/action_log.py
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, NoReturn

from wumpus_engine.config.schema import SandboxLevel
from wumpus_engine.core.errors import (
    InvalidWorldError,
    LogFormatError,
    LogStateError,
    LogViewError,
)
from wumpus_engine.core.results import RunInfo, StepEvent, TerminationReason
from wumpus_engine.core.rules import SolvabilityPolicy
from wumpus_engine.core.world import WorldSpec
from wumpus_sdk.actions import Action
from wumpus_sdk.dto import ObservationDTO, Outcome
from wumpus_sdk.errors import InvalidActionError, ProtocolError
from wumpus_sdk.percept import Percept
from wumpus_sdk.types import Direction, Position


class LogView(StrEnum):
    '''Who a log view is for.'''

    OBSERVER = 'OBSERVER'  # full detail; instructor and Observer tools only
    AGENT_SAFE = 'AGENT_SAFE'  # only what the agent saw and chose; safe to give to students


@dataclass(frozen=True, slots=True)
class HiddenDetail:
    '''Present only in the OBSERVER view.'''

    death_cause: TerminationReason | None  # DIED_PIT or DIED_WUMPUS
    shot_path: tuple[Position, ...]
    killed_wumpus_at: Position | None


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    '''One turn.'''

    turn: int  # index of the observation answered; the first is 0
    observation: ObservationDTO  # exactly what the agent saw before choosing
    action: Action  # the executed action
    events: tuple[StepEvent, ...]
    score_delta: int
    score_total: int
    result_percept: Percept | None  # percept after the action; None on a terminal step
    hidden: HiddenDetail | None = None  # OBSERVER view only


@dataclass(frozen=True, slots=True)
class LogHeader:
    '''Game-level data written once.'''

    framework_version: str
    protocol_version: int
    run: RunInfo  # holds world_seed: OBSERVER view only
    generator_version: str  # OBSERVER view only
    config_fingerprint: str  # OBSERVER view only
    world_layout: WorldSpec | None  # OBSERVER view only; lets replay work without the generator
    score_rules: dict[str, int]
    max_actions: int
    initial_arrows: int


@dataclass(frozen=True, slots=True)
class LogFooter:
    '''Game-level result written once.'''

    termination: TerminationReason  # OBSERVER view only
    outcome: Outcome
    final_score: int
    actions_taken: int
    fault_message: str | None  # OBSERVER view only; sanitized
    wall_time_s: float  # OBSERVER view only


_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {'kind', 'turn', 'observation', 'action', 'events', 'score_delta', 'score_total', 'result_percept'}
)
_OBSERVATION_KEYS: Final[frozenset[str]] = frozenset(
    {'step_index', 'position', 'direction', 'arrows_remaining', 'has_gold', 'percept'}
)
_HIDDEN_KEYS: Final[frozenset[str]] = frozenset({'death_cause', 'shot_path', 'killed_wumpus_at'})
_RUN_KEYS: Final[frozenset[str]] = frozenset(
    {'run_id', 'agent_id', 'scenario_id', 'world_seed', 'agent_seed', 'isolation'}
)
_OBSERVER_HEADER_KEYS: Final[frozenset[str]] = frozenset(
    {
        'kind', 'view', 'framework_version', 'protocol_version', 'run', 'generator_version',
        'config_fingerprint', 'world_layout', 'score_rules', 'max_actions', 'initial_arrows',
    }
)
_SAFE_HEADER_KEYS: Final[frozenset[str]] = frozenset(
    {
        'kind', 'view', 'framework_version', 'protocol_version', 'agent_id', 'run_id', 'score_rules',
        'max_actions', 'initial_arrows',
    }
)
_OBSERVER_FOOTER_KEYS: Final[frozenset[str]] = frozenset(
    {'kind', 'termination', 'outcome', 'final_score', 'actions_taken', 'fault_message', 'wall_time_s'}
)
_SAFE_FOOTER_KEYS: Final[frozenset[str]] = frozenset({'kind', 'outcome', 'final_score', 'actions_taken'})


# ---- strict JSON readers (every failure is a LogFormatError) -------------------------------------
def _as_obj(raw: object, what: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise LogFormatError(f'{what} must be a JSON object')
    return {str(key): value for key, value in raw.items()}


def _as_list(raw: object, what: str) -> list[object]:
    if not isinstance(raw, list):
        raise LogFormatError(f'{what} must be a JSON array')
    return list(raw)


def _as_int(raw: object, what: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise LogFormatError(f'{what} must be an integer')
    return raw


def _as_float(raw: object, what: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise LogFormatError(f'{what} must be a number')
    return float(raw)


def _as_bool(raw: object, what: str) -> bool:
    if not isinstance(raw, bool):
        raise LogFormatError(f'{what} must be true or false')
    return raw


def _as_str(raw: object, what: str) -> str:
    if not isinstance(raw, str):
        raise LogFormatError(f'{what} must be a string')
    return raw


def _opt_str(raw: object, what: str) -> str | None:
    return None if raw is None else _as_str(raw, what)


def _check_keys(
    data: dict[str, object], required: frozenset[str], optional: frozenset[str], what: str
) -> None:
    missing = required - data.keys()
    unknown = data.keys() - required - optional
    if missing:
        raise LogFormatError(f'{what} is missing {sorted(missing)}')
    if unknown:
        raise LogFormatError(f'{what} has unknown keys {sorted(unknown)}')


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LogFormatError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def _reject_constant(name: str) -> NoReturn:
    raise LogFormatError(f'{name} is not allowed in a log')


def _dumps(value: object) -> str:
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def _parse_line(line: str, number: int) -> dict[str, object]:
    try:
        value = json.loads(line, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise LogFormatError(f'line {number} is not valid JSON: {exc}') from exc
    return _as_obj(value, f'line {number}')


def _read_records(path: Path) -> list[dict[str, object]]:
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError as exc:
        raise LogFormatError(f'{path} is not valid UTF-8') from exc
    if not text.endswith('\n'):
        raise LogFormatError('a log file must end with a newline')
    records = [_parse_line(line, number) for number, line in enumerate(text.split('\n')[:-1], start=1)]
    if len(records) < 2:
        raise LogFormatError('a log file needs a header line and a footer line')
    return records


# ---- entries --------------------------------------------------------------------------------------
def _observation_to_json(obs: ObservationDTO) -> dict[str, object]:
    return {
        'step_index': obs.step_index,
        'position': obs.position.to_wire(),
        'direction': obs.direction.value,
        'arrows_remaining': obs.arrows_remaining,
        'has_gold': obs.has_gold,
        'percept': obs.percept.to_wire(),
    }


def _observation_from_json(raw: object) -> ObservationDTO:
    data = _as_obj(raw, 'observation')
    _check_keys(data, _OBSERVATION_KEYS, frozenset(), 'observation')
    try:
        return ObservationDTO(
            step_index=_as_int(data['step_index'], 'observation.step_index'),
            position=Position.from_wire(data['position']),
            direction=Direction(_as_str(data['direction'], 'observation.direction')),
            arrows_remaining=_as_int(data['arrows_remaining'], 'observation.arrows_remaining'),
            has_gold=_as_bool(data['has_gold'], 'observation.has_gold'),
            percept=Percept.from_wire(data['percept']),
        )
    except (ValueError, ProtocolError) as exc:
        raise LogFormatError(f'bad observation: {exc}') from exc


def _positions_from_json(raw: object, what: str) -> tuple[Position, ...]:
    try:
        return tuple(Position.from_wire(item) for item in _as_list(raw, what))
    except ProtocolError as exc:
        raise LogFormatError(f'bad {what}: {exc}') from exc


def _hidden_to_json(hidden: HiddenDetail) -> dict[str, object]:
    return {
        'death_cause': None if hidden.death_cause is None else hidden.death_cause.value,
        'shot_path': [cell.to_wire() for cell in hidden.shot_path],
        'killed_wumpus_at': None if hidden.killed_wumpus_at is None else hidden.killed_wumpus_at.to_wire(),
    }


def _hidden_from_json(raw: object) -> HiddenDetail:
    data = _as_obj(raw, 'hidden')
    _check_keys(data, _HIDDEN_KEYS, frozenset(), 'hidden')
    cause = _opt_str(data['death_cause'], 'hidden.death_cause')
    killed = data['killed_wumpus_at']
    try:
        return HiddenDetail(
            death_cause=None if cause is None else TerminationReason(cause),
            shot_path=_positions_from_json(data['shot_path'], 'hidden.shot_path'),
            killed_wumpus_at=None if killed is None else Position.from_wire(killed),
        )
    except (ValueError, ProtocolError) as exc:
        raise LogFormatError(f'bad hidden detail: {exc}') from exc


def _entry_to_json(entry: ActionLogEntry, view: LogView) -> dict[str, object]:
    out: dict[str, object] = {
        'kind': 'entry',
        'turn': entry.turn,
        'observation': _observation_to_json(entry.observation),
        'action': entry.action.value,
        'events': [event.value for event in entry.events],
        'score_delta': entry.score_delta,
        'score_total': entry.score_total,
        'result_percept': None if entry.result_percept is None else entry.result_percept.to_wire(),
    }
    if view is LogView.OBSERVER and entry.hidden is not None:
        out['hidden'] = _hidden_to_json(entry.hidden)
    return out


def _entry_from_json(raw: dict[str, object], *, observer: bool) -> ActionLogEntry:
    _check_keys(raw, _ENTRY_KEYS, frozenset({'hidden'}) if observer else frozenset(), 'entry')
    if raw['kind'] != 'entry':
        raise LogFormatError('expected an entry line')
    percept = raw['result_percept']
    hidden = raw.get('hidden')
    try:
        return ActionLogEntry(
            turn=_as_int(raw['turn'], 'entry.turn'),
            observation=_observation_from_json(raw['observation']),
            action=Action.parse(_as_str(raw['action'], 'entry.action')),
            events=tuple(StepEvent(_as_str(item, 'entry.events')) for item in _as_list(raw['events'], 'entry.events')),
            score_delta=_as_int(raw['score_delta'], 'entry.score_delta'),
            score_total=_as_int(raw['score_total'], 'entry.score_total'),
            result_percept=None if percept is None else Percept.from_wire(percept),
            hidden=None if hidden is None else _hidden_from_json(hidden),
        )
    except (ValueError, ProtocolError, InvalidActionError) as exc:
        raise LogFormatError(f'bad entry: {exc}') from exc


# ---- headers and footers --------------------------------------------------------------------------
def _score_rules_from_json(raw: object) -> dict[str, int]:
    data = _as_obj(raw, 'score_rules')
    return {key: _as_int(value, f'score_rules.{key}') for key, value in data.items()}


def _observer_header_json(header: LogHeader) -> dict[str, object]:
    run = header.run
    return {
        'view': LogView.OBSERVER.value,
        'framework_version': header.framework_version,
        'protocol_version': header.protocol_version,
        'run': {
            'run_id': run.run_id,
            'agent_id': run.agent_id,
            'scenario_id': run.scenario_id,
            'world_seed': run.world_seed,
            'agent_seed': run.agent_seed,
            'isolation': run.isolation.value,
        },
        'generator_version': header.generator_version,
        'config_fingerprint': header.config_fingerprint,
        'world_layout': None if header.world_layout is None else header.world_layout.to_json(),
        'score_rules': dict(header.score_rules),
        'max_actions': header.max_actions,
        'initial_arrows': header.initial_arrows,
    }


def _safe_header_json(header: LogHeader, reveal_seed: bool) -> dict[str, object]:
    out: dict[str, object] = {
        'view': LogView.AGENT_SAFE.value,
        'framework_version': header.framework_version,
        'protocol_version': header.protocol_version,
        'agent_id': header.run.agent_id,
        'run_id': header.run.run_id,
        'score_rules': dict(header.score_rules),
        'max_actions': header.max_actions,
        'initial_arrows': header.initial_arrows,
    }
    if reveal_seed:
        out['world_seed'] = header.run.world_seed
    return out


def _observer_header_from_json(data: dict[str, object]) -> LogHeader:
    _check_keys(data, _OBSERVER_HEADER_KEYS, frozenset(), 'header')
    run = _as_obj(data['run'], 'header.run')
    _check_keys(run, _RUN_KEYS, frozenset(), 'header.run')
    layout = data['world_layout']
    try:
        info = RunInfo(
            run_id=_as_str(run['run_id'], 'run.run_id'),
            agent_id=_as_str(run['agent_id'], 'run.agent_id'),
            scenario_id=_as_str(run['scenario_id'], 'run.scenario_id'),
            world_seed=_as_int(run['world_seed'], 'run.world_seed'),
            agent_seed=_as_int(run['agent_seed'], 'run.agent_seed'),
            isolation=SandboxLevel(_as_str(run['isolation'], 'run.isolation')),
        )
        world_layout = None if layout is None else WorldSpec.from_json(_as_obj(layout, 'header.world_layout'))
        if world_layout is not None:
            world_layout.validate(SolvabilityPolicy.NONE)  # a replayable layout must satisfy I1 to I4
    except (ValueError, InvalidWorldError) as exc:
        raise LogFormatError(f'bad header: {exc}') from exc
    return LogHeader(
        framework_version=_as_str(data['framework_version'], 'header.framework_version'),
        protocol_version=_as_int(data['protocol_version'], 'header.protocol_version'),
        run=info,
        generator_version=_as_str(data['generator_version'], 'header.generator_version'),
        config_fingerprint=_as_str(data['config_fingerprint'], 'header.config_fingerprint'),
        world_layout=world_layout,
        score_rules=_score_rules_from_json(data['score_rules']),
        max_actions=_as_int(data['max_actions'], 'header.max_actions'),
        initial_arrows=_as_int(data['initial_arrows'], 'header.initial_arrows'),
    )


def _safe_header_from_json(data: dict[str, object]) -> dict[str, object]:
    _check_keys(data, _SAFE_HEADER_KEYS, frozenset({'world_seed'}), 'header')
    _as_str(data['framework_version'], 'header.framework_version')
    _as_int(data['protocol_version'], 'header.protocol_version')
    _as_str(data['agent_id'], 'header.agent_id')
    _as_str(data['run_id'], 'header.run_id')
    _score_rules_from_json(data['score_rules'])
    _as_int(data['max_actions'], 'header.max_actions')
    _as_int(data['initial_arrows'], 'header.initial_arrows')
    if 'world_seed' in data:
        _as_int(data['world_seed'], 'header.world_seed')
    return {key: value for key, value in data.items() if key != 'kind'}


def _observer_footer_json(footer: LogFooter) -> dict[str, object]:
    return {
        'termination': footer.termination.value,
        'outcome': footer.outcome.value,
        'final_score': footer.final_score,
        'actions_taken': footer.actions_taken,
        'fault_message': footer.fault_message,
        'wall_time_s': footer.wall_time_s,
    }


def _safe_footer_json(footer: LogFooter) -> dict[str, object]:
    return {
        'outcome': footer.outcome.value,
        'final_score': footer.final_score,
        'actions_taken': footer.actions_taken,
    }


def _observer_footer_from_json(data: dict[str, object]) -> LogFooter:
    _check_keys(data, _OBSERVER_FOOTER_KEYS, frozenset(), 'footer')
    try:
        return LogFooter(
            termination=TerminationReason(_as_str(data['termination'], 'footer.termination')),
            outcome=Outcome(_as_str(data['outcome'], 'footer.outcome')),
            final_score=_as_int(data['final_score'], 'footer.final_score'),
            actions_taken=_as_int(data['actions_taken'], 'footer.actions_taken'),
            fault_message=_opt_str(data['fault_message'], 'footer.fault_message'),
            wall_time_s=_as_float(data['wall_time_s'], 'footer.wall_time_s'),
        )
    except ValueError as exc:
        raise LogFormatError(f'bad footer: {exc}') from exc


def _safe_footer_from_json(data: dict[str, object]) -> dict[str, object]:
    _check_keys(data, _SAFE_FOOTER_KEYS, frozenset(), 'footer')
    try:
        Outcome(_as_str(data['outcome'], 'footer.outcome'))
    except ValueError as exc:
        raise LogFormatError(f'bad footer: {exc}') from exc
    _as_int(data['final_score'], 'footer.final_score')
    _as_int(data['actions_taken'], 'footer.actions_taken')
    return {key: value for key, value in data.items() if key != 'kind'}


class ActionLog:
    '''One game's turn records, served in an OBSERVER view and an AGENT_SAFE view.'''

    def __init__(
        self,
        header: LogHeader,
        *,
        reveal_seed_in_agent_safe: bool = False,
        max_entries: int = 100001,
    ) -> None:
        '''Pre: max_entries >= 1. reveal_seed_in_agent_safe may be True only for PRACTICE scenarios.
        Post: open and empty. Raises: ValueError: max_entries < 1.
        '''
        if max_entries < 1:
            raise ValueError('max_entries must be >= 1')
        self._header: LogHeader | None = header
        self._safe_header: dict[str, object] | None = None  # set only for logs loaded from AGENT_SAFE files
        self._reveal = reveal_seed_in_agent_safe
        self._max_entries = max_entries
        self._entries: list[ActionLogEntry] = []
        self._footer: LogFooter | None = None
        self._safe_footer: dict[str, object] | None = None
        self._closed = False
        self._source = LogView.OBSERVER

    @property
    def is_closed(self) -> bool:
        '''Post: True once close() has been called (or the log was loaded).'''
        return self._closed

    @property
    def source_view(self) -> LogView:
        '''Post: OBSERVER for a log built here or loaded from an OBSERVER file; AGENT_SAFE for a log loaded
        from an AGENT_SAFE file.
        '''
        return self._source

    def append(self, entry: ActionLogEntry) -> None:
        '''Pre: not closed; entry.turn equals the number of entries so far; the log is not full.
        Post: the entry is stored. Raises: LogStateError: closed, wrong turn number, or log full.
        '''
        if self._closed:
            raise LogStateError('the log is closed')
        if entry.turn != len(self._entries):
            raise LogStateError(f'expected turn {len(self._entries)}, got {entry.turn}')
        if len(self._entries) >= self._max_entries:
            raise LogStateError('the log is full')
        self._entries.append(entry)

    def close(self, footer: LogFooter) -> None:
        '''Pre: not closed. Post: closed; append() now raises. Raises: LogStateError: already closed.'''
        if self._closed:
            raise LogStateError('the log is already closed')
        self._footer = footer
        self._closed = True

    def _require_view(self, view: LogView) -> None:
        '''Raises: LogViewError: OBSERVER requested from a log loaded from an AGENT_SAFE file.'''
        if view is LogView.OBSERVER and self._source is LogView.AGENT_SAFE:
            raise LogViewError('an AGENT_SAFE log cannot be upgraded to the OBSERVER view')

    def header(self, view: LogView) -> dict[str, object]:
        '''Pre: none. Post: JSON-safe header. OBSERVER: every LogHeader field. AGENT_SAFE: framework_version,
        protocol_version, agent_id, run_id, score_rules, max_actions, initial_arrows, and world_seed only if
        reveal_seed_in_agent_safe is set. Raises: LogViewError: OBSERVER from an AGENT_SAFE-loaded log.
        '''
        self._require_view(view)
        if view is LogView.AGENT_SAFE and self._safe_header is not None:
            return dict(self._safe_header)
        if self._header is None:
            raise LogViewError('this log holds no OBSERVER header')
        if view is LogView.OBSERVER:
            return _observer_header_json(self._header)
        return _safe_header_json(self._header, self._reveal)

    def entries(self, view: LogView) -> tuple[ActionLogEntry, ...]:
        '''Pre: none. Post: for AGENT_SAFE every entry has hidden set to None; otherwise identical.
        Raises: LogViewError: OBSERVER from an AGENT_SAFE-loaded log.
        '''
        self._require_view(view)
        if view is LogView.OBSERVER:
            return tuple(self._entries)
        return tuple(entry if entry.hidden is None else replace(entry, hidden=None) for entry in self._entries)

    def footer(self, view: LogView) -> dict[str, object] | None:
        '''Pre: none. Post: None until closed. AGENT_SAFE holds only outcome, final_score and actions_taken.
        Raises: LogViewError: OBSERVER from an AGENT_SAFE-loaded log.
        '''
        self._require_view(view)
        if not self._closed:
            return None
        if view is LogView.AGENT_SAFE and self._safe_footer is not None:
            return dict(self._safe_footer)
        if self._footer is None:
            raise LogViewError('this log holds no OBSERVER footer')
        if view is LogView.OBSERVER:
            return _observer_footer_json(self._footer)
        return _safe_footer_json(self._footer)

    def to_jsonl(self, view: LogView) -> str:
        '''Pre: the log is closed. Post: pure; ASCII JSON, one object per line (header, one entry per turn,
        footer), every line ending with a newline. Raises: LogStateError: not closed. LogViewError: as for
        header().
        '''
        if not self._closed:
            raise LogStateError('the log is not closed yet')
        header = self.header(view)
        footer = self.footer(view)
        if footer is None:
            raise LogStateError('the log is not closed yet')
        lines = [_dumps({'kind': 'header', **header})]
        lines.extend(_dumps(_entry_to_json(entry, view)) for entry in self.entries(view))
        lines.append(_dumps({'kind': 'footer', **footer}))
        return '\n'.join(lines) + '\n'

    def save(self, path: Path, view: LogView, *, overwrite: bool = False) -> Path:
        '''Pre: the log is closed; the parent directory exists. Post: the file holds exactly to_jsonl(view);
        returns `path`. Raises: LogStateError: not closed. LogViewError. FileExistsError: exists and not
        overwrite.
        '''
        text = self.to_jsonl(view)
        with path.open('w' if overwrite else 'x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        return path

    @classmethod
    def load(cls, path: Path) -> ActionLog:
        '''Pre: `path` is a file written by save(). Post: a closed log whose source_view is the view the file
        was saved with; a log loaded from an OBSERVER file never reveals the seed in its AGENT_SAFE view.
        Raises: LogFormatError: not valid JSON Lines, wrong shape, an invalid world layout, or turns not
        0, 1, 2, ...
        '''
        records = _read_records(path)
        head, body, tail = records[0], records[1:-1], records[-1]
        if head.get('kind') != 'header' or tail.get('kind') != 'footer':
            raise LogFormatError('a log file starts with a header line and ends with a footer line')
        try:
            view = LogView(_as_str(head.get('view'), 'header.view'))
        except ValueError as exc:
            raise LogFormatError(f'bad header view: {exc}') from exc
        observer = view is LogView.OBSERVER
        entries = [_entry_from_json(record, observer=observer) for record in body]
        if [entry.turn for entry in entries] != list(range(len(entries))):
            raise LogFormatError('entry turns must run 0, 1, 2, ... without gaps')
        if observer:
            log = cls(_observer_header_from_json(head), max_entries=max(len(entries), 1))
            log._footer = _observer_footer_from_json(tail)
        else:
            log = cls.__new__(cls)
            log._header = None
            log._safe_header = _safe_header_from_json(head)
            log._reveal = 'world_seed' in log._safe_header
            log._max_entries = max(len(entries), 1)
            log._footer = None
            log._safe_footer = _safe_footer_from_json(tail)
            log._source = LogView.AGENT_SAFE
        log._entries = entries
        log._closed = True
        return log