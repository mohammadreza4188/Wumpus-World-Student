# wumpus_engine/config/config_manager.py
from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Final, TypeVar

from wumpus_engine.config.schema import (
    AgentSpec,
    ConfigIssue,
    DisplayConfig,
    EngineLimits,
    EvaluationSuite,
    ExplicitLayout,
    FrameworkConfig,
    PathsConfig,
    SandboxLevel,
    SandboxPolicy,
    ScenarioConfig,
    ScenarioVisibility,
)
from wumpus_engine.core.errors import (
    ConfigNotFoundError,
    ConfigSecurityError,
    ConfigValidationError,
    InvalidWorldError,
    SuiteValidationError,
)
from wumpus_engine.core.rules import DEFAULT_SCORE_TABLE, ScoreTable, SolvabilityPolicy
from wumpus_engine.core.world import WorldSpec
from wumpus_sdk.errors import ProtocolError
from wumpus_sdk.types import MAX_WUMPUSES, MIN_WUMPUSES, Position

_T = TypeVar('_T')
_E = TypeVar('_E', bound=StrEnum)

_MAX_FILE_BYTES: Final[int] = 1 << 20  # a configuration file larger than 1 MiB is refused
_MAX_ACTIONS_LIMIT: Final[int] = 100000
_GRID_CELLS: Final[int] = 400
_SCORE_LIMIT: Final[int] = 100000
_TRAVERSAL_CHARS: Final[str] = '/\\\x00:'

# fullmatch is used everywhere: unlike '$', it never accepts a trailing newline.
_NAME_RE: Final[re.Pattern[str]] = re.compile(r'[a-z0-9_]{1,40}')
_AGENT_ID_RE: Final[re.Pattern[str]] = re.compile(r'[a-z0-9][a-z0-9_-]{0,39}')
_ENTRY_POINT_RE: Final[re.Pattern[str]] = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*')
_ENV_NAME_RE: Final[re.Pattern[str]] = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_IMAGE_RE: Final[re.Pattern[str]] = re.compile(r'[A-Za-z0-9][A-Za-z0-9._/:@-]{0,255}')

_FRAMEWORK_KEYS: Final[frozenset[str]] = frozenset({'engine', 'scoring', 'sandbox', 'paths', 'display'})
_ENGINE_KEYS: Final[frozenset[str]] = frozenset(
    {
        'max_actions', 'action_timeout_s', 'startup_timeout_s', 'game_wall_clock_s',
        'max_invalid_actions', 'max_frame_bytes',
    }
)
_SCORING_KEYS: Final[frozenset[str]] = frozenset(
    {'action', 'shoot_extra', 'gold_grab', 'wumpus_kill', 'death', 'forfeit'}
)
_SANDBOX_KEYS: Final[frozenset[str]] = frozenset(
    {
        'level', 'memory_limit_mb', 'cpu_time_limit_s', 'max_processes', 'max_open_files',
        'max_output_bytes', 'workdir_size_mb', 'env_allowlist', 'python_executable',
        'container_image', 'allow_network', 'allow_unsandboxed_ranking',
    }
)
_PATHS_KEYS: Final[frozenset[str]] = frozenset({'results_dir', 'scenarios_dir', 'suites_dir'})
_DISPLAY_KEYS: Final[frozenset[str]] = frozenset({'renderer', 'color', 'step_delay_s'})
_AGENTS_KEYS: Final[frozenset[str]] = frozenset({'agents'})
_AGENT_KEYS: Final[frozenset[str]] = frozenset({'agent_id', 'entry_point', 'source_dir', 'display_name'})
_SCENARIO_KEYS: Final[frozenset[str]] = frozenset(
    {
        'scenario_id', 'difficulty_rank', 'visibility', 'pit_count', 'wumpus_count', 'seed', 'layout',
        'solvability', 'max_actions',
    }
)
_LAYOUT_KEYS: Final[frozenset[str]] = frozenset({'pits', 'wumpuses', 'gold'})
_SUITE_KEYS: Final[frozenset[str]] = frozenset({'name', 'official', 'scenarios'})


class _FileProblem(Exception):
    '''A configuration file is too large, not UTF-8 or not valid TOML.'''

    def __init__(self, label: str, problem: str) -> None:
        super().__init__(f'{label}: {problem}')
        self.label = label
        self.problem = problem


class _Issues:
    '''Collects ConfigIssue records.'''

    def __init__(self) -> None:
        self.items: list[ConfigIssue] = []

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.items if item.severity == 'ERROR')

    def add(self, severity: str, path: str, message: str) -> None:
        self.items.append(ConfigIssue(severity=severity, path=path, message=message))


def _range_text(ge: float | None, le: float | None) -> str:
    if ge is not None and le is not None:
        return f'between {ge:g} and {le:g}'
    if ge is not None:
        return f'at least {ge:g}'
    return f'at most {le:g}'


class _Table:
    '''A parsed TOML table. Every accessor records an issue instead of raising.'''

    def __init__(self, data: dict[str, object], label: str, prefix: str, issues: _Issues) -> None:
        self.data = data
        self.label = label
        self.prefix = prefix
        self.issues = issues

    def where(self, key: str) -> str:
        '''Post: 'file: table.key' (or just the file when both are empty).'''
        name = f'{self.prefix}{key}'.rstrip('.')
        return f'{self.label}: {name}' if name else self.label

    def error(self, key: str, message: str) -> None:
        self.issues.add('ERROR', self.where(key), message)

    def warning(self, key: str, message: str) -> None:
        self.issues.add('WARNING', self.where(key), message)

    def unknown(self, allowed: frozenset[str]) -> None:
        '''Rule C8: unknown keys are errors.'''
        for key in sorted(set(self.data) - allowed):
            self.error(key, 'unknown key')

    def table(self, key: str) -> _Table | None:
        if key not in self.data:
            return None
        value = self.data[key]
        if not isinstance(value, dict):
            self.error(key, 'must be a table')
            return None
        return _Table({str(k): v for k, v in value.items()}, self.label, f'{self.prefix}{key}.', self.issues)

    # ---- integers
    def _int(self, key: str, ge: int | None, le: int | None, rule: str | None) -> int | None:
        if key not in self.data:
            return None
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            self.error(key, f'must be an integer, got {type(value).__name__}')
            return None
        if (ge is not None and value < ge) or (le is not None and value > le):
            prefix = f'{rule}: ' if rule else ''
            self.error(key, f'{prefix}must be {_range_text(ge, le)}, got {value}')
            return None
        return value

    def integer(self, key: str, default: int, *, ge: int | None = None, le: int | None = None) -> int:
        value = self._int(key, ge, le, None)
        return default if value is None else value

    def optional_integer(
        self, key: str, *, ge: int | None = None, le: int | None = None, rule: str | None = None
    ) -> int | None:
        return self._int(key, ge, le, rule)

    def required_integer(
        self, key: str, *, ge: int | None = None, le: int | None = None, rule: str | None = None
    ) -> int | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        return self._int(key, ge, le, rule)

    # ---- numbers
    def number(
        self,
        key: str,
        default: float,
        *,
        gt: float | None = None,
        ge: float | None = None,
        le: float | None = None,
    ) -> float:
        if key not in self.data:
            return default
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(key, f'must be a number, got {type(value).__name__}')
            return default
        try:
            number = float(value)
        except OverflowError:
            self.error(key, 'is out of range')
            return default
        if not math.isfinite(number):
            self.error(key, 'must be a finite number')
            return default
        if gt is not None and not number > gt:
            self.error(key, f'must be greater than {gt:g}, got {number:g}')
            return default
        if (ge is not None and number < ge) or (le is not None and number > le):
            self.error(key, f'must be {_range_text(ge, le)}, got {number:g}')
            return default
        return number

    # ---- booleans
    def boolean(self, key: str, default: bool) -> bool:
        if key not in self.data:
            return default
        value = self.data[key]
        if not isinstance(value, bool):
            self.error(key, f'must be true or false, got {type(value).__name__}')
            return default
        return value

    def required_boolean(self, key: str) -> bool | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        value = self.data[key]
        if not isinstance(value, bool):
            self.error(key, f'must be true or false, got {type(value).__name__}')
            return None
        return value

    # ---- strings
    def _str(self, key: str, pattern: re.Pattern[str] | None) -> str | None:
        if key not in self.data:
            return None
        value = self.data[key]
        if not isinstance(value, str):
            self.error(key, f'must be a string, got {type(value).__name__}')
            return None
        if not value or '\x00' in value:
            self.error(key, 'must be a non-empty string without NUL characters')
            return None
        if pattern is not None and not pattern.fullmatch(value):
            self.error(key, f'must match {pattern.pattern}, got {value!r}')
            return None
        return value

    def optional_string(self, key: str, pattern: re.Pattern[str] | None = None) -> str | None:
        return self._str(key, pattern)

    def required_string(self, key: str, pattern: re.Pattern[str] | None = None) -> str | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        return self._str(key, pattern)

    def choice(self, key: str, allowed: tuple[str, ...], default: str) -> str:
        value = self._str(key, None)
        if value is None:
            return default
        if value not in allowed:
            self.error(key, f'must be one of {", ".join(allowed)}, got {value!r}')
            return default
        return value

    def _enum(self, key: str, enum_type: type[_E]) -> _E | None:
        value = self._str(key, None)
        if value is None:
            return None
        try:
            return enum_type(value)
        except ValueError:
            allowed = ', '.join(member.value for member in enum_type)
            self.error(key, f'must be one of {allowed}, got {value!r}')
            return None

    def enum(self, key: str, enum_type: type[_E], default: _E) -> _E:
        value = self._enum(key, enum_type)
        return default if value is None else value

    def required_enum(self, key: str, enum_type: type[_E]) -> _E | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        return self._enum(key, enum_type)

    # ---- lists
    def _strings(self, key: str, pattern: re.Pattern[str]) -> tuple[str, ...] | None:
        value = self.data[key]
        if not isinstance(value, list):
            self.error(key, 'must be an array of strings')
            return None
        items: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not pattern.fullmatch(item):
                self.error(f'{key}[{index}]', f'must be a string matching {pattern.pattern}')
                return None
            items.append(item)
        return tuple(items)

    def string_list(
        self, key: str, default: tuple[str, ...], pattern: re.Pattern[str]
    ) -> tuple[str, ...]:
        if key not in self.data:
            return default
        items = self._strings(key, pattern)
        return default if items is None else items

    def required_string_list(self, key: str, pattern: re.Pattern[str]) -> tuple[str, ...] | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        return self._strings(key, pattern)

    # ---- cells
    def position(self, key: str) -> Position | None:
        if key not in self.data:
            self.error(key, 'is required')
            return None
        try:
            return Position.from_wire(self.data[key])
        except ProtocolError as exc:
            self.error(key, str(exc))
            return None

    def positions(self, key: str, *, required: bool) -> tuple[Position, ...] | None:
        if key not in self.data:
            if required:
                self.error(key, 'is required')
                return None
            return ()
        raw = self.data[key]
        if not isinstance(raw, list):
            self.error(key, 'must be an array of [x, y] pairs')
            return None
        cells: list[Position] = []
        for index, item in enumerate(raw):
            try:
                cells.append(Position.from_wire(item))
            except ProtocolError as exc:
                self.error(f'{key}[{index}]', str(exc))
        return tuple(cells) if len(cells) == len(raw) else None


# ---- framework.toml ---------------------------------------------------------------------------------
def _parse_engine(table: _Table | None) -> EngineLimits:
    defaults = EngineLimits()
    if table is None:
        return defaults
    table.unknown(_ENGINE_KEYS)
    return EngineLimits(
        max_actions=table.integer('max_actions', defaults.max_actions, ge=1, le=_MAX_ACTIONS_LIMIT),
        action_timeout_s=table.number('action_timeout_s', defaults.action_timeout_s, gt=0.0, le=3600.0),
        startup_timeout_s=table.number('startup_timeout_s', defaults.startup_timeout_s, gt=0.0, le=3600.0),
        game_wall_clock_s=table.number('game_wall_clock_s', defaults.game_wall_clock_s, gt=0.0, le=86400.0),
        max_invalid_actions=table.integer('max_invalid_actions', defaults.max_invalid_actions, ge=0, le=100),
        max_frame_bytes=table.integer('max_frame_bytes', defaults.max_frame_bytes, ge=1024, le=1_048_576),
    )


def _parse_scoring(table: _Table | None) -> ScoreTable:
    defaults = DEFAULT_SCORE_TABLE
    if table is None:
        return defaults
    table.unknown(_SCORING_KEYS)
    low, high = -_SCORE_LIMIT, _SCORE_LIMIT
    try:
        score = ScoreTable(
            action=table.integer('action', defaults.action, ge=low, le=high),
            shoot_extra=table.integer('shoot_extra', defaults.shoot_extra, ge=low, le=high),
            gold_grab=table.integer('gold_grab', defaults.gold_grab, ge=low, le=high),
            wumpus_kill=table.integer('wumpus_kill', defaults.wumpus_kill, ge=low, le=high),
            death=table.integer('death', defaults.death, ge=low, le=high),
            forfeit=table.integer('forfeit', defaults.forfeit, ge=low, le=high),
        )
    except ValueError as exc:
        table.error('', str(exc))
        return defaults
    if score != defaults:
        table.warning('', 'C4: the score table differs from the spec defaults (an ERROR in an official suite)')
    return score


def _parse_sandbox(table: _Table | None) -> SandboxPolicy:
    defaults = SandboxPolicy()
    if table is None:
        return defaults
    table.unknown(_SANDBOX_KEYS)
    level = table.enum('level', SandboxLevel, defaults.level)
    allow_network = table.boolean('allow_network', defaults.allow_network)
    if level is SandboxLevel.OS_SANDBOX and allow_network:
        table.error('allow_network', 'C5: level OS_SANDBOX requires allow_network = false')
    return SandboxPolicy(
        level=level,
        memory_limit_mb=table.integer('memory_limit_mb', defaults.memory_limit_mb, ge=16, le=65536),
        cpu_time_limit_s=table.integer('cpu_time_limit_s', defaults.cpu_time_limit_s, ge=1, le=86400),
        max_processes=table.integer('max_processes', defaults.max_processes, ge=1, le=4096),
        max_open_files=table.integer('max_open_files', defaults.max_open_files, ge=8, le=65536),
        max_output_bytes=table.integer('max_output_bytes', defaults.max_output_bytes, ge=1024, le=1 << 30),
        workdir_size_mb=table.integer('workdir_size_mb', defaults.workdir_size_mb, ge=1, le=4096),
        env_allowlist=table.string_list('env_allowlist', defaults.env_allowlist, _ENV_NAME_RE),
        python_executable=table.optional_string('python_executable'),
        container_image=table.optional_string('container_image', _IMAGE_RE),
        allow_network=allow_network,
        allow_unsandboxed_ranking=table.boolean('allow_unsandboxed_ranking', defaults.allow_unsandboxed_ranking),
    )


def _parse_display(table: _Table | None) -> DisplayConfig:
    defaults = DisplayConfig()
    if table is None:
        return defaults
    table.unknown(_DISPLAY_KEYS)
    return DisplayConfig(
        renderer=table.choice('renderer', ('null', 'text'), defaults.renderer),
        color=table.boolean('color', defaults.color),
        step_delay_s=table.number('step_delay_s', defaults.step_delay_s, ge=0.0, le=60.0),
    )


# ---- scenarios and suites -----------------------------------------------------------------------------
def _parse_layout(table: _Table, solvability: SolvabilityPolicy) -> ExplicitLayout | None:
    '''Rule C2: an explicit layout must pass WorldSpec.validate(solvability).'''
    table.unknown(_LAYOUT_KEYS)
    pits = table.positions('pits', required=False)
    wumpuses = table.positions('wumpuses', required=True)
    gold = table.position('gold')
    if pits is None or wumpuses is None or gold is None:
        return None
    if len(set(pits)) != len(pits) or len(set(wumpuses)) != len(wumpuses):
        table.error('', 'C2: a layout list names the same cell twice')
        return None
    try:
        WorldSpec(pits=frozenset(pits), wumpuses=frozenset(wumpuses), gold=gold).validate(solvability)
    except InvalidWorldError as exc:
        table.error('', f'C2: {exc}')
        return None
    return ExplicitLayout(pits=pits, wumpuses=wumpuses, gold=gold)


def _parse_scenario(
    data: dict[str, object], label: str, directory: ScenarioVisibility, stem: str
) -> tuple[ScenarioConfig | None, list[ConfigIssue]]:
    '''Post: a ScenarioConfig when no error was found (rules C1, C2, C8, C9), plus every issue.'''
    issues = _Issues()
    table = _Table(data, label, '', issues)
    table.unknown(_SCENARIO_KEYS)
    scenario_id = table.required_string('scenario_id', _NAME_RE)
    if scenario_id is not None and scenario_id != stem:
        table.error('scenario_id', f'must equal the file name {stem!r}, got {scenario_id!r}')
    rank = table.required_integer('difficulty_rank', ge=1, le=1000)
    visibility = table.required_enum('visibility', ScenarioVisibility)
    if visibility is not None and visibility is not directory:
        table.error(
            'visibility',
            f'C9: {visibility.value} does not match the directory scenarios/{directory.value.lower()}',
        )
    solvability = table.enum('solvability', SolvabilityPolicy, SolvabilityPolicy.REACHABLE_WITH_SHOOTING)
    max_actions = table.optional_integer('max_actions', ge=1, le=_MAX_ACTIONS_LIMIT)
    layout_table = table.table('layout')
    layout: ExplicitLayout | None = None
    seed: int | None = None
    pit_count: int | None
    wumpus_count: int | None
    if layout_table is not None:
        layout = _parse_layout(layout_table, solvability)
        if 'seed' in data:
            table.error('seed', 'C2: must be absent when a [layout] is given')
        given_pits = table.optional_integer('pit_count', ge=0, le=_GRID_CELLS)
        given_wumpuses = table.optional_integer('wumpus_count', ge=0, le=_GRID_CELLS)
        pit_count = wumpus_count = None
        if layout is not None:
            pit_count, wumpus_count = len(layout.pits), len(layout.wumpuses)
            if given_pits is not None and given_pits != pit_count:
                table.error('pit_count', f'C2: {given_pits} does not match the layout, which has {pit_count} pits')
            if given_wumpuses is not None and given_wumpuses != wumpus_count:
                table.error(
                    'wumpus_count',
                    f'C2: {given_wumpuses} does not match the layout, which has {wumpus_count} Wumpuses',
                )
    else:
        pit_count = table.required_integer('pit_count', ge=0, le=_GRID_CELLS, rule='C1')
        wumpus_count = table.required_integer('wumpus_count', ge=MIN_WUMPUSES, le=MAX_WUMPUSES, rule='C1')
        seed = table.optional_integer('seed', ge=0, le=(1 << 63) - 1)
        if pit_count is not None and wumpus_count is not None and pit_count + wumpus_count + 2 > _GRID_CELLS:
            table.error('pit_count', f'C1: pit_count + wumpus_count + 2 must be at most {_GRID_CELLS}')
    if (
        issues.error_count
        or scenario_id is None
        or rank is None
        or visibility is None
        or pit_count is None
        or wumpus_count is None
    ):
        return None, issues.items
    scenario = ScenarioConfig(
        scenario_id=scenario_id,
        difficulty_rank=rank,
        visibility=visibility,
        pit_count=pit_count,
        wumpus_count=wumpus_count,
        seed=seed,
        layout=layout,
        solvability=solvability,
        max_actions=max_actions,
    )
    return scenario, issues.items


def _parse_suite(
    data: dict[str, object],
    label: str,
    name: str,
    resolve: Callable[[str], ScenarioConfig | None],
    scoring: ScoreTable | None,
) -> tuple[EvaluationSuite | None, list[ConfigIssue]]:
    '''Post: an EvaluationSuite whose scenarios are ordered by difficulty_rank and that passed validate()
    (rule C3); in an official suite a non-default score table is an error (rule C4).
    '''
    issues = _Issues()
    table = _Table(data, label, '', issues)
    table.unknown(_SUITE_KEYS)
    declared = table.required_string('name')
    if declared is not None and declared != name:
        table.error('name', f'must equal the file name {name!r}, got {declared!r}')
    official = table.required_boolean('official')
    ids = table.required_string_list('scenarios', _NAME_RE)
    if ids is not None and len(set(ids)) != len(ids):
        table.error('scenarios', 'C3: scenario ids must be unique')
    if official and scoring is not None and scoring != DEFAULT_SCORE_TABLE:
        table.error('official', 'C4: an official suite requires the spec score table')
    scenarios: list[ScenarioConfig] = []
    if ids is not None:
        for scenario_id in dict.fromkeys(ids):
            found = resolve(scenario_id)
            if found is None:
                table.error('scenarios', f'unknown scenario {scenario_id!r}')
            else:
                scenarios.append(found)
    if issues.error_count or official is None or ids is None:
        return None, issues.items
    suite = EvaluationSuite(
        name=name,
        official=official,
        scenarios=tuple(sorted(scenarios, key=lambda scenario: scenario.difficulty_rank)),
    )
    try:
        suite.validate()
    except SuiteValidationError as exc:
        table.error('', f'C3: {exc}')
        return None, issues.items
    return suite, issues.items


# ---- helpers -----------------------------------------------------------------------------------------
def _is_within(child: Path, root: Path) -> bool:
    return child == root or root in child.parents


def _resolve(path: Path, what: str) -> Path:
    '''Raises: ConfigSecurityError: the path cannot be resolved (for example a symlink loop).'''
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ConfigSecurityError(f'{what}: {path} cannot be resolved: {exc}') from exc


def _check_name(value: str, what: str) -> None:
    '''Raises: ConfigSecurityError: the name could leave its directory. ConfigValidationError: invalid name.'''
    if any(ch in value for ch in _TRAVERSAL_CHARS) or '..' in value or value.startswith(('.', '~')):
        raise ConfigSecurityError(f'{what} {value!r} tries to leave its directory')
    if not _NAME_RE.fullmatch(value):
        message = f'{what} {value!r} must match {_NAME_RE.pattern}'
        raise ConfigValidationError(message, (message,))


def _valid_entry_point(value: str) -> bool:
    '''Rule C7: the contract regex, plus every dotted module part must be an identifier.'''
    if not _ENTRY_POINT_RE.fullmatch(value):
        return False
    return all(part.isidentifier() for part in value.split(':', 1)[0].split('.'))


def _raise_if_errors(label: str, issues: list[ConfigIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == 'ERROR']
    if errors:
        lines = tuple(f'{issue.path}: {issue.message}' for issue in errors)
        raise ConfigValidationError(f'{label}: {len(errors)} configuration error(s); first: {lines[0]}', lines)


def _require(value: _T | None, label: str) -> _T:
    if value is None:
        message = f'{label}: the file could not be loaded'
        raise ConfigValidationError(message, (message,))
    return value


def _sorted_issues(issues: list[ConfigIssue]) -> list[ConfigIssue]:
    return sorted(issues, key=lambda issue: (issue.severity != 'ERROR', issue.path, issue.message))


def _canonical(value: object) -> object:
    '''Post: parsed TOML content in a form whose JSON text ignores key order (and 2 versus 2.0).'''
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return str(value)


class ConfigManager:
    '''Loads and validates every settings file and returns immutable configuration objects.'''

    def __init__(self, config_dir: Path) -> None:
        '''Pre: config_dir is an existing directory. Post: nothing is read yet.
        Raises: ConfigNotFoundError: config_dir is missing or not a directory.
        '''
        if not config_dir.is_dir():
            raise ConfigNotFoundError(f'configuration directory {config_dir} is missing or not a directory')
        self._root = _resolve(config_dir, 'config_dir')

    # ---- public API ----------------------------------------------------------------------------------
    def load_framework(self) -> FrameworkConfig:
        '''Pre: framework.toml exists. Post: a validated, immutable FrameworkConfig; missing optional keys take
        the dataclass defaults; unknown keys are errors; relative [paths] values stay relative to the
        configuration directory.
        Raises: ConfigNotFoundError: file missing. ConfigValidationError: a rule C1 to C9 is violated.
        ConfigSecurityError: a path escapes the allowed roots.
        '''
        label, data = self._load_raw('framework.toml')
        framework, issues = self._parse_framework(data, label)
        _raise_if_errors(label, issues)
        return _require(framework, label)

    def load_agents(self) -> tuple[AgentSpec, ...]:
        '''Pre: agents.toml exists. Post: agent_ids are unique; every entry_point and source_dir passed rules
        C6 and C7; each source_dir is the resolved absolute path (relative values are anchored at the
        configuration directory); no graded scenario file lies under any source_dir.
        Raises: ConfigNotFoundError, ConfigValidationError, ConfigSecurityError.
        '''
        label, data = self._load_raw('agents.toml')
        agents, issues = self._parse_agents(data, label)
        _raise_if_errors(label, issues)
        specs = _require(agents, label)
        self._check_graded_hidden(specs)
        return specs

    def list_scenarios(self) -> tuple[str, ...]:
        '''Pre: none. Post: all valid scenario_ids under scenarios/practice and scenarios/graded, sorted; a
        listing by file name only, no file is parsed.
        Raises: ConfigNotFoundError: the scenarios directory is missing.
        '''
        if not (self._root / 'scenarios').is_dir():
            raise ConfigNotFoundError('the scenarios directory is missing')
        names = {
            path.stem
            for visibility in ScenarioVisibility
            for path in self._scenario_files(visibility)
            if _NAME_RE.fullmatch(path.stem)
        }
        return tuple(sorted(names))

    def load_scenario(self, scenario_id: str) -> ScenarioConfig:
        '''Pre: scenario_id matches [a-z0-9_]{1,40}. Post: a validated ScenarioConfig; an explicit layout has
        already passed WorldSpec.validate(scenario.solvability) and its counts are consistent.
        Raises: ConfigNotFoundError: no such scenario. ConfigValidationError: a rule C1, C2, C8 or C9 is
        violated. ConfigSecurityError: the id or a path tries to leave the scenarios directory.
        '''
        _check_name(scenario_id, 'scenario id')
        if not (self._root / 'scenarios').is_dir():
            raise ConfigNotFoundError('the scenarios directory is missing')
        found = [
            visibility
            for visibility in ScenarioVisibility
            if self._scenario_path(visibility, scenario_id).is_file()
            or self._scenario_path(visibility, scenario_id).is_symlink()
        ]
        if not found:
            raise ConfigNotFoundError(f'no scenario {scenario_id!r} under scenarios/practice or scenarios/graded')
        if len(found) > 1:
            message = f'scenario id {scenario_id!r} exists in both scenarios/practice and scenarios/graded'
            raise ConfigValidationError(message, (message,))
        visibility = found[0]
        label, data = self._load_raw('scenarios', visibility.value.lower(), f'{scenario_id}.toml')
        scenario, issues = _parse_scenario(data, label, visibility, scenario_id)
        _raise_if_errors(label, issues)
        return _require(scenario, label)

    def load_suite(self, suite_name: str) -> EvaluationSuite:
        '''Pre: suite_name matches [a-z0-9_]{1,40}. Post: every named scenario is loaded; scenarios are ordered
        by difficulty_rank and suite.validate() has already passed.
        Raises: ConfigNotFoundError, ConfigValidationError (rule C3, and C4 for official suites),
        ConfigSecurityError.
        '''
        _check_name(suite_name, 'suite name')
        label, data = self._load_raw('suites', f'{suite_name}.toml')
        scoring = self.load_framework().scoring if data.get('official') is True else None
        suite, issues = _parse_suite(data, label, suite_name, self.load_scenario, scoring)
        _raise_if_errors(label, issues)
        return _require(suite, label)

    def validate_all(self) -> tuple[ConfigIssue, ...]:
        '''Pre: none. Post: an empty tuple means the configuration is valid; otherwise all issues, errors
        first. Nothing stops at the first problem.
        Raises: ConfigNotFoundError: config_dir vanished.
        '''
        issues, _ = self._scan()
        return tuple(_sorted_issues(issues))

    def config_fingerprint(self) -> str:
        '''Pre: none. Post: sha256 hex over the canonical JSON of the parsed content of every configuration
        file; the same settings give the same digest regardless of key order, comments or whitespace.
        Raises: ConfigValidationError: the configuration cannot be loaded.
        '''
        try:
            issues, raws = self._scan()
        except ConfigNotFoundError as exc:
            raise ConfigValidationError(str(exc), (str(exc),)) from exc
        errors = tuple(f'{issue.path}: {issue.message}' for issue in issues if issue.severity == 'ERROR')
        if errors:
            raise ConfigValidationError('the configuration cannot be loaded', errors)
        canonical = json.dumps(_canonical(raws), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    # ---- file access ---------------------------------------------------------------------------------
    def _locate(self, parts: tuple[str, ...]) -> tuple[str, Path]:
        '''Post: (label, resolved path) of a regular file inside the configuration directory.
        Raises: ConfigSecurityError: the file resolves outside it (rule C6). ConfigNotFoundError: missing.
        '''
        label = '/'.join(parts)
        resolved = _resolve(self._root.joinpath(*parts), label)
        if not _is_within(resolved, self._root):
            raise ConfigSecurityError(f'{label} resolves to {resolved}, outside the configuration directory')
        if not resolved.is_file():
            raise ConfigNotFoundError(f'{label} does not exist')
        return label, resolved

    def _read_raw(self, *parts: str) -> tuple[str, dict[str, object]]:
        '''Post: (label, parsed TOML table); tomllib is the only parser used.
        Raises: ConfigNotFoundError, ConfigSecurityError, _FileProblem (too large, not UTF-8, not TOML).
        '''
        label, path = self._locate(parts)
        try:
            with path.open('rb') as handle:
                payload = handle.read(_MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise ConfigNotFoundError(f'{label} cannot be read: {exc.strerror or exc}') from exc
        if len(payload) > _MAX_FILE_BYTES:
            raise _FileProblem(label, f'is larger than {_MAX_FILE_BYTES} bytes')
        try:
            parsed = tomllib.loads(payload.decode('utf-8'))
        except UnicodeDecodeError as exc:
            raise _FileProblem(label, 'is not valid UTF-8') from exc
        except (tomllib.TOMLDecodeError, RecursionError) as exc:
            raise _FileProblem(label, f'is not valid TOML: {exc}') from exc
        return label, parsed

    def _load_raw(self, *parts: str) -> tuple[str, dict[str, object]]:
        '''As _read_raw(), but a _FileProblem becomes a ConfigValidationError.'''
        try:
            return self._read_raw(*parts)
        except _FileProblem as exc:
            raise ConfigValidationError(str(exc), (str(exc),)) from exc

    def _anchor(self, raw: str, what: str) -> Path:
        '''Rule C6: a relative path must stay inside the configuration directory after symlink resolution;
        an absolute path is an explicitly listed root. Post: the resolved absolute path.
        Raises: ConfigSecurityError.
        '''
        path = Path(raw)
        if path.is_absolute():
            return _resolve(path, what)
        resolved = _resolve(self._root / path, what)
        if not _is_within(resolved, self._root):
            raise ConfigSecurityError(f'{what}: {raw!r} resolves to {resolved}, outside the configuration directory')
        return resolved

    def _scenario_path(self, visibility: ScenarioVisibility, scenario_id: str) -> Path:
        return self._root / 'scenarios' / visibility.value.lower() / f'{scenario_id}.toml'

    def _scenario_files(self, visibility: ScenarioVisibility) -> list[Path]:
        directory = self._root / 'scenarios' / visibility.value.lower()
        if not directory.is_dir():
            return []
        return sorted(path for path in directory.iterdir() if path.suffix == '.toml' and not path.is_dir())

    # ---- parsing -------------------------------------------------------------------------------------
    def _parse_framework(
        self, data: dict[str, object], label: str
    ) -> tuple[FrameworkConfig | None, list[ConfigIssue]]:
        issues = _Issues()
        root = _Table(data, label, '', issues)
        root.unknown(_FRAMEWORK_KEYS)
        config = FrameworkConfig(
            engine=_parse_engine(root.table('engine')),
            scoring=_parse_scoring(root.table('scoring')),
            sandbox=_parse_sandbox(root.table('sandbox')),
            paths=self._parse_paths(root.table('paths')),
            display=_parse_display(root.table('display')),
        )
        return (None if issues.error_count else config), issues.items

    def _parse_paths(self, table: _Table | None) -> PathsConfig:
        defaults = PathsConfig()
        if table is None:
            return defaults
        table.unknown(_PATHS_KEYS)
        chosen: dict[str, Path] = {}
        fixed = {'scenarios_dir': self._root / 'scenarios', 'suites_dir': self._root / 'suites'}
        for key, default in (
            ('results_dir', defaults.results_dir),
            ('scenarios_dir', defaults.scenarios_dir),
            ('suites_dir', defaults.suites_dir),
        ):
            raw = table.optional_string(key)
            if raw is None:
                chosen[key] = default
                continue
            resolved = self._anchor(raw, table.where(key))  # rule C6
            chosen[key] = Path(raw)
            if key in fixed and resolved != _resolve(fixed[key], key):
                table.warning(key, f'ignored: ConfigManager always reads {fixed[key].name}/ under the configuration directory')
        return PathsConfig(
            results_dir=chosen['results_dir'],
            scenarios_dir=chosen['scenarios_dir'],
            suites_dir=chosen['suites_dir'],
        )

    def _parse_agents(
        self, data: dict[str, object], label: str
    ) -> tuple[tuple[AgentSpec, ...] | None, list[ConfigIssue]]:
        issues = _Issues()
        root = _Table(data, label, '', issues)
        root.unknown(_AGENTS_KEYS)
        entries = data.get('agents', [])
        if not isinstance(entries, list):
            root.error('agents', 'must be an array of tables ([[agents]])')
            return None, issues.items
        specs: list[AgentSpec] = []
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                root.error(f'agents[{index}]', 'must be a table')
                continue
            table = _Table({str(k): v for k, v in entry.items()}, label, f'agents[{index}].', issues)
            before = issues.error_count
            table.unknown(_AGENT_KEYS)
            agent_id = table.required_string('agent_id', _AGENT_ID_RE)
            if agent_id is not None:
                if agent_id in seen:
                    table.error('agent_id', f'duplicate agent_id {agent_id!r}')
                seen.add(agent_id)
            entry_point = table.required_string('entry_point')
            if entry_point is not None and not _valid_entry_point(entry_point):
                table.error('entry_point', f'must look like package.module:ClassName, got {entry_point!r}')
            raw_dir = table.required_string('source_dir')
            source_dir = None if raw_dir is None else self._anchor(raw_dir, table.where('source_dir'))
            display_name = table.optional_string('display_name')
            if display_name is not None and (len(display_name) > 64 or not display_name.isprintable()):
                table.error('display_name', 'must be printable text of at most 64 characters')
            if (
                issues.error_count == before
                and agent_id is not None
                and entry_point is not None
                and source_dir is not None
            ):
                specs.append(
                    AgentSpec(
                        agent_id=agent_id,
                        entry_point=entry_point,
                        source_dir=source_dir,
                        display_name=display_name,
                    )
                )
        return (None if issues.error_count else tuple(specs)), issues.items

    def _check_graded_hidden(self, agents: tuple[AgentSpec, ...]) -> None:
        '''Rule C6: no graded scenario file may lie under any agent's source_dir.
        Raises: ConfigSecurityError.
        '''
        graded = _resolve(self._root / 'scenarios' / 'graded', 'scenarios/graded')
        secret = [graded]
        secret.extend(_resolve(path, path.name) for path in self._scenario_files(ScenarioVisibility.GRADED))
        for spec in agents:
            for path in secret:
                if _is_within(path, spec.source_dir):
                    raise ConfigSecurityError(
                        f'agent {spec.agent_id!r}: source_dir {spec.source_dir} contains graded scenario data {path}'
                    )

    # ---- whole-configuration scan --------------------------------------------------------------------
    def _scan_file(
        self,
        parts: tuple[str, ...],
        raws: dict[str, object],
        issues: list[ConfigIssue],
        parse: Callable[[dict[str, object], str], tuple[_T | None, list[ConfigIssue]]],
    ) -> _T | None:
        '''Read and parse one file, recording every problem as an issue instead of raising.'''
        label = '/'.join(parts)
        try:
            label, data = self._read_raw(*parts)
            raws[label] = data
            value, found = parse(data, label)
        except _FileProblem as exc:
            issues.append(ConfigIssue('ERROR', exc.label, exc.problem))
            return None
        except (ConfigNotFoundError, ConfigSecurityError) as exc:
            issues.append(ConfigIssue('ERROR', label, str(exc)))
            return None
        issues.extend(found)
        return value

    def _scan(self) -> tuple[list[ConfigIssue], dict[str, object]]:
        '''Post: every issue found in framework.toml, agents.toml, all scenarios and all suites, plus the
        parsed content of each file that could be read. Raises: ConfigNotFoundError: config_dir vanished.
        '''
        if not self._root.is_dir():
            raise ConfigNotFoundError(f'configuration directory {self._root} no longer exists')
        issues: list[ConfigIssue] = []
        raws: dict[str, object] = {}
        framework = self._scan_file(('framework.toml',), raws, issues, self._parse_framework)
        agents = self._scan_file(('agents.toml',), raws, issues, self._parse_agents)
        if agents is not None:
            try:
                self._check_graded_hidden(agents)
            except ConfigSecurityError as exc:
                issues.append(ConfigIssue('ERROR', 'agents.toml', str(exc)))
        scenarios: dict[str, ScenarioConfig] = {}
        if not (self._root / 'scenarios').is_dir():
            issues.append(ConfigIssue('ERROR', 'scenarios', 'the scenarios directory is missing'))
        owners: dict[str, str] = {}
        for visibility in ScenarioVisibility:
            directory = visibility.value.lower()
            for path in self._scenario_files(visibility):
                label = f'scenarios/{directory}/{path.name}'
                if not _NAME_RE.fullmatch(path.stem):
                    issues.append(ConfigIssue('ERROR', label, f'the file name is not a valid scenario id ({_NAME_RE.pattern})'))
                    continue
                if path.stem in owners:
                    issues.append(
                        ConfigIssue('ERROR', label, f'scenario id {path.stem!r} is also used in scenarios/{owners[path.stem]}')
                    )
                    continue
                owners[path.stem] = directory
                found = self._scan_file(
                    ('scenarios', directory, path.name),
                    raws,
                    issues,
                    partial(_parse_scenario, directory=visibility, stem=path.stem),
                )
                if found is not None:
                    scenarios[found.scenario_id] = found
        suites_dir = self._root / 'suites'
        if suites_dir.is_dir():
            scoring = None if framework is None else framework.scoring
            known: dict[str, ScenarioConfig] = scenarios
            for path in sorted(p for p in suites_dir.iterdir() if p.suffix == '.toml' and not p.is_dir()):
                if not _NAME_RE.fullmatch(path.stem):
                    issues.append(ConfigIssue('ERROR', f'suites/{path.name}', f'the file name is not a valid suite name ({_NAME_RE.pattern})'))
                    continue
                self._scan_file(
                    ('suites', path.name),
                    raws,
                    issues,
                    partial(_parse_suite, name=path.stem, resolve=known.get, scoring=scoring),
                )
        return issues, raws