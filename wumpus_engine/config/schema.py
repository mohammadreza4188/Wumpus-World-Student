# wumpus_engine/config/schema.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# ScoreTable and SolvabilityPolicy are defined in core/rules.py and only used here.
from wumpus_engine.core.errors import SuiteValidationError
from wumpus_engine.core.rules import DEFAULT_SCORE_TABLE, ScoreTable, SolvabilityPolicy
from wumpus_sdk.types import Position


class SandboxLevel(StrEnum):
    '''How strongly the agent process is isolated.'''

    PROCESS = 'PROCESS'  # separate interpreter, exec not fork, scrubbed env, rlimits; NOT rank-eligible
    OS_SANDBOX = 'OS_SANDBOX'  # PROCESS plus container or separate OS user: no network, engine invisible


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    '''Limits and isolation settings for one agent process.'''

    level: SandboxLevel = SandboxLevel.OS_SANDBOX
    memory_limit_mb: int = 512
    cpu_time_limit_s: int = 600
    max_processes: int = 1
    max_open_files: int = 32
    max_output_bytes: int = 1_048_576  # cap on captured stderr
    workdir_size_mb: int = 16  # ephemeral tmpfs working directory
    env_allowlist: tuple[str, ...] = ('PATH', 'LANG')
    python_executable: str | None = None  # interpreter of the agent-side virtualenv or image
    container_image: str | None = None
    allow_network: bool = False  # must stay False when level is OS_SANDBOX
    allow_unsandboxed_ranking: bool = False


@dataclass(frozen=True, slots=True)
class EngineLimits:
    '''Per-game engine limits.'''

    max_actions: int = 2000
    action_timeout_s: float = 2.0
    startup_timeout_s: float = 10.0
    game_wall_clock_s: float = 300.0
    max_invalid_actions: int = 3
    max_frame_bytes: int = 65536


@dataclass(frozen=True, slots=True)
class PathsConfig:
    '''Directories used by the framework.'''

    results_dir: Path = Path('results')
    scenarios_dir: Path = Path('scenarios')
    suites_dir: Path = Path('suites')


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    '''Observer display settings.'''

    renderer: str = 'null'  # 'null' or 'text'
    color: bool = False
    step_delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FrameworkConfig:
    '''Everything framework.toml configures.'''

    engine: EngineLimits = EngineLimits()
    scoring: ScoreTable = DEFAULT_SCORE_TABLE
    sandbox: SandboxPolicy = SandboxPolicy()
    paths: PathsConfig = PathsConfig()
    display: DisplayConfig = DisplayConfig()


class ScenarioVisibility(StrEnum):
    '''Who may see a scenario file and its seed.'''

    PRACTICE = 'PRACTICE'  # students may see the file and its seed
    GRADED = 'GRADED'  # instructor only


@dataclass(frozen=True, slots=True)
class ExplicitLayout:
    '''A hand-placed map.'''

    pits: tuple[Position, ...]
    wumpuses: tuple[Position, ...]
    gold: Position


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    '''One scenario: generated from a seed or hand-placed.'''

    scenario_id: str
    difficulty_rank: int  # 1 is the easiest
    visibility: ScenarioVisibility
    pit_count: int
    wumpus_count: int
    seed: int | None = None  # None: drawn at run time and recorded in RunInfo
    layout: ExplicitLayout | None = None  # hand-placed map; makes seed irrelevant
    solvability: SolvabilityPolicy = SolvabilityPolicy.REACHABLE_WITH_SHOOTING
    max_actions: int | None = None  # overrides EngineLimits.max_actions


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    '''An ordered set of scenarios an agent is evaluated on.'''

    name: str
    official: bool
    scenarios: tuple[ScenarioConfig, ...]

    def validate(self) -> None:
        '''Pre: none.
        Post: returns only if the suite is not empty, scenario ids are unique, difficulty_rank is strictly
        increasing and, when official, 5 <= len(scenarios) <= 6 and every scenario is GRADED.
        Raises: SuiteValidationError.
        '''
        if not self.scenarios:
            raise SuiteValidationError(f'suite {self.name!r} has no scenarios')
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(ids)) != len(ids):
            raise SuiteValidationError(f'suite {self.name!r} repeats a scenario id')
        ranks = [scenario.difficulty_rank for scenario in self.scenarios]
        if any(later <= earlier for earlier, later in zip(ranks, ranks[1:])):
            raise SuiteValidationError(f'suite {self.name!r}: difficulty_rank must strictly increase')
        if self.official:
            if not 5 <= len(self.scenarios) <= 6:
                raise SuiteValidationError(
                    f'official suite {self.name!r} needs 5 to 6 scenarios, has {len(self.scenarios)}'
                )
            if any(s.visibility is not ScenarioVisibility.GRADED for s in self.scenarios):
                raise SuiteValidationError(f'official suite {self.name!r} may only hold GRADED scenarios')


@dataclass(frozen=True, slots=True)
class AgentSpec:
    '''One student agent to run.'''

    agent_id: str
    entry_point: str  # 'package.module:ClassName'
    source_dir: Path | None = None # copied read-only into the sandbox
    display_name: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    '''One problem found while validating the configuration.'''

    severity: str  # 'ERROR' or 'WARNING'
    path: str  # file and key, for example 'scenarios/graded/s3.toml: pit_count'
    message: str