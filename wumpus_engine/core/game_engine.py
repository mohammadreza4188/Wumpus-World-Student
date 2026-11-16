# wumpus_engine/core/game_engine.py
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Final, assert_never

from wumpus_engine.config.schema import EngineLimits, ScenarioConfig, ScenarioVisibility
from wumpus_engine.core.action_log import (
    ActionLog,
    ActionLogEntry,
    HiddenDetail,
    LogFooter,
    LogHeader,
)
from wumpus_engine.core.agent_state import AgentState
from wumpus_engine.core.errors import (
    AgentCrashedError,
    AgentFault,
    AgentLaunchError,
    AgentReportedError,
    AgentTimeoutError,
    EngineStateError,
    ProtocolViolationError,
)
from wumpus_engine.core.results import (
    FinalScore,
    GameResult,
    GameStatus,
    GatewayReport,
    RunInfo,
    StepEvent,
    StepResult,
    TerminationReason,
)
from wumpus_engine.core.rules import ScoreEvent, ScoreTable
from wumpus_engine.core.score_manager import ScoreManager
from wumpus_engine.core.world import ShotResult, World, WorldSpec
from wumpus_engine.core.world_generator import WorldGenerator
from wumpus_engine.display.renderer import NullRenderer, Renderer, RenderFrame, RenderMode
from wumpus_engine.gateway.agent_gateway import AgentPort
from wumpus_engine.gateway.sandbox import sanitize_text
from wumpus_sdk.actions import Action
from wumpus_sdk.dto import (
    PROTOCOL_VERSION,
    ActionDTO,
    GameOverDTO,
    InvalidActionDTO,
    ObservationDTO,
    SessionInitDTO,
)
from wumpus_sdk.errors import InvalidActionError
from wumpus_sdk.percept import Percept
from wumpus_sdk.types import GRID_HEIGHT, GRID_WIDTH, START_DIRECTION, START_POSITION

_FRAMEWORK_VERSION: Final[str] = '0.1.0'
_MAX_AGENT_NAME_CHARS: Final[int] = 64
_MAX_MESSAGE_CHARS: Final[int] = 1000
_RENDER_MODE: Final[RenderMode] = RenderMode.OBSERVER

# Terminations that record a FORFEIT score event (spec 4.2.2). ENGINE_ERROR records none.
_FORFEIT_REASONS: Final[frozenset[TerminationReason]] = frozenset(
    {
        TerminationReason.ACTION_LIMIT,
        TerminationReason.AGENT_TIMEOUT,
        TerminationReason.AGENT_CRASH,
        TerminationReason.AGENT_ERROR,
        TerminationReason.PROTOCOL_VIOLATION,
        TerminationReason.INVALID_ACTION_LIMIT,
    }
)


def _score_rules(table: ScoreTable) -> dict[str, int]:
    '''Post: the public price list sent in SessionInitDTO.score_rules and written to the log header.'''
    return {
        'action': table.action,
        'shoot_extra': table.shoot_extra,
        'gold_grab': table.gold_grab,
        'wumpus_kill': table.wumpus_kill,
        'death': table.death,
        'forfeit': table.forfeit,
    }


def _fingerprint(
    limits: EngineLimits, max_actions: int, rules: dict[str, int], scenario: ScenarioConfig
) -> str:
    '''Post: sha256 of the effective per-game configuration known to the engine.'''
    payload = {
        'limits': asdict(limits),
        'max_actions': max_actions,
        'scoring': rules,
        'solvability': scenario.solvability.value,
        'pit_count': scenario.pit_count,
        'wumpus_count': scenario.wumpus_count,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('ascii')).hexdigest()


class GameEngine:
    '''Runs exactly one game: one world, one agent session, one score ledger.'''

    def __init__(
        self,
        scenario: ScenarioConfig,
        run_info: RunInfo,
        limits: EngineLimits,
        score_table: ScoreTable,
        gateway: AgentPort,
        renderer: Renderer,
        *,
        generator: WorldGenerator | None = None,
        world: World | None = None,
    ) -> None:
        '''Pre: `gateway` is fresh. A given `world` is used as is and `generator` is ignored; otherwise the
        world is generated from (scenario, run_info.world_seed). No I/O on `gateway` or `renderer`.
        Post: status READY; agent on (1, 1) facing EAST with one arrow per Wumpus; empty ScoreManager;
        ActionLog header written.
        Raises: WorldGenerationError: no valid world. InvalidWorldError: an explicit layout violates I1-I5.
        '''
        generator_version = 'injected'
        if world is None:
            maker = generator if generator is not None else WorldGenerator()
            world = World(maker.generate(scenario, run_info.world_seed))
            generator_version = maker.GENERATOR_VERSION
        self._scenario = scenario
        self._run_info = run_info
        self._limits = limits
        self._gateway = gateway
        self._renderer = renderer
        self._world = world
        self._initial_arrows = world.wumpuses_alive
        self._agent = AgentState(arrows=self._initial_arrows)
        self._score = ScoreManager(score_table)
        self._max_actions = (
            scenario.max_actions if scenario.max_actions is not None else limits.max_actions
        )
        self._invalid_strikes = 0
        self._status = GameStatus.READY
        self._percept: Percept = world.perceive(START_POSITION)
        self._observation: ObservationDTO | None = None
        self._termination: TerminationReason | None = None
        self._final: FinalScore | None = None
        self._fault_message: str | None = None
        self._wall_time_s = 0.0
        self._t0 = time.monotonic()
        rules = _score_rules(score_table)
        snapshot = world.snapshot()
        layout = WorldSpec(
            pits=snapshot.pits,
            wumpuses=snapshot.wumpuses_alive | snapshot.wumpuses_dead,
            gold=snapshot.gold_position,
        )
        header = LogHeader(
            framework_version=_FRAMEWORK_VERSION,
            protocol_version=PROTOCOL_VERSION,
            run=run_info,
            generator_version=generator_version,
            config_fingerprint=_fingerprint(limits, self._max_actions, rules, scenario),
            world_layout=layout,
            score_rules=rules,
            max_actions=self._max_actions,
            initial_arrows=self._initial_arrows,
        )
        self._log = ActionLog(
            header,
            reveal_seed_in_agent_safe=scenario.visibility is ScenarioVisibility.PRACTICE,
            max_entries=self._max_actions + 1,
        )

    @property
    def status(self) -> GameStatus:
        '''Post: READY, RUNNING or FINISHED.'''
        return self._status

    @property
    def action_log(self) -> ActionLog:
        '''Post: the game's log; complete and closed once status is FINISHED.'''
        return self._log

    def build_session_init(self) -> SessionInitDTO:
        '''Pre: none. Post: public rules only; no hidden data. Raises: nothing.'''
        return SessionInitDTO(
            protocol_version=PROTOCOL_VERSION,
            grid_width=GRID_WIDTH,
            grid_height=GRID_HEIGHT,
            start_position=START_POSITION,
            start_direction=START_DIRECTION,
            initial_arrows=self._initial_arrows,
            max_actions=self._max_actions,
            action_timeout_s=self._limits.action_timeout_s,
            score_rules=_score_rules(self._score.table),
            agent_seed=self._run_info.agent_seed,
        )

    def build_observation(self) -> ObservationDTO:
        '''Pre: status READY or RUNNING (the last step was not terminal).
        Post: step_index == actions_taken; the percept carries the transient flags of the last action.
        The same object is returned until the next action is applied.
        Raises: EngineStateError: the game is over.
        '''
        if self._status is GameStatus.FINISHED:
            raise EngineStateError('the game is over')
        if self._observation is None:
            agent = self._agent
            self._observation = ObservationDTO(
                step_index=agent.actions_taken,
                position=agent.position,
                direction=agent.direction,
                arrows_remaining=agent.arrows,
                has_gold=agent.has_gold,
                percept=self._percept,
            )
        return self._observation

    def apply_action(self, action: Action) -> StepResult:
        '''Pre: status READY or RUNNING (READY becomes RUNNING).
        Post: executes the rule table of spec 4.2.1; actions_taken grows by one; score events are recorded in
        table order; a terminal step records DEATH or FORFEIT, closes the log and sets status FINISHED.
        Raises: EngineStateError: the game already ended.
        '''
        if self._status is GameStatus.FINISHED:
            raise EngineStateError('the game already ended')
        self._status = GameStatus.RUNNING
        agent, world, score = self._agent, self._world, self._score
        observation = self.build_observation()  # exactly what the agent saw before choosing
        turn = agent.actions_taken
        score.record(ScoreEvent.ACTION, step_index=turn)  # step 1: every action costs its price
        agent.actions_taken = turn + 1
        events: list[StepEvent] = []
        termination: TerminationReason | None = None
        death_cause: TerminationReason | None = None
        shot: ShotResult | None = None
        bump = False
        scream = False
        match action:  # step 2: the action's row of the rule table
            case Action.TURN_LEFT:
                agent.turn_left()
                events.append(StepEvent.TURNED)
            case Action.TURN_RIGHT:
                agent.turn_right()
                events.append(StepEvent.TURNED)
            case Action.MOVE_FORWARD:
                target = agent.position.neighbor(agent.direction)
                if target is None:
                    bump = True
                    events.append(StepEvent.BUMPED)
                else:
                    agent.move_to(target)
                    events.append(StepEvent.MOVED)
                    if world.is_pit(target):
                        death_cause = TerminationReason.DIED_PIT
                    elif world.is_living_wumpus(target):
                        death_cause = TerminationReason.DIED_WUMPUS
                    if death_cause is not None:
                        agent.die()
                        score.record(ScoreEvent.DEATH, step_index=turn)
                        events.append(StepEvent.DIED)
                        termination = death_cause
            case Action.SHOOT:
                if agent.arrows == 0:
                    events.append(StepEvent.NO_ARROWS)
                else:
                    agent.consume_arrow()
                    score.record(ScoreEvent.SHOOT_ARROW, step_index=turn)
                    shot = world.fire_arrow(agent.position, agent.direction)
                    events.append(StepEvent.ARROW_FIRED)
                    if shot.hit:
                        score.record(ScoreEvent.WUMPUS_KILL, step_index=turn)
                        scream = True
                        events.append(StepEvent.WUMPUS_KILLED)
                    else:
                        events.append(StepEvent.ARROW_MISSED)
            case Action.GRAB:
                if world.take_gold(agent.position):
                    agent.pick_up_gold()
                    score.record(ScoreEvent.GOLD_GRAB, step_index=turn)
                    events.append(StepEvent.GOLD_GRABBED)
                else:
                    events.append(StepEvent.NOTHING_TO_GRAB)
            case Action.CLIMB:
                if agent.position == START_POSITION:
                    agent.exit_cave()
                    termination = (
                        TerminationReason.CLIMBED_WITH_GOLD
                        if agent.has_gold
                        else TerminationReason.CLIMBED_WITHOUT_GOLD
                    )
                    events.append(StepEvent.CLIMBED)
                else:
                    events.append(StepEvent.CLIMB_REJECTED)
            case _:
                assert_never(action)
        if termination is None and agent.actions_taken >= self._max_actions:  # step 3
            termination = TerminationReason.ACTION_LIMIT
            score.record(ScoreEvent.FORFEIT, step_index=turn)
        percept: Percept | None = None
        if termination is None:  # step 4: the game continues
            percept = world.perceive(agent.position, bump=bump, scream=scream)
            self._percept = percept
            self._observation = None
        hidden: HiddenDetail | None = None
        if shot is not None or death_cause is not None:
            hidden = HiddenDetail(
                death_cause=death_cause,
                shot_path=() if shot is None else shot.path,
                killed_wumpus_at=None if shot is None else shot.killed_at,
            )
        delta = score.delta_for_step(turn)
        total = score.total
        self._log_append(
            ActionLogEntry(
                turn=turn,
                observation=observation,
                action=action,
                events=tuple(events),
                score_delta=delta,
                score_total=total,
                result_percept=percept,
                hidden=hidden,
            )
        )  # step 5
        if termination is not None:
            self._finish(termination, None)
        return StepResult(
            turn=turn,
            action=action,
            events=tuple(events),
            percept=percept,
            score_delta=delta,
            score_total=total,
            terminal=termination is not None,
            termination=termination,
        )

    def run(self) -> GameResult:
        '''Pre: status READY.
        Post: status FINISHED; gateway.close() was called exactly once, even on failure; renderer.end() was
        called once; the score is finalized and the log closed. Every agent fault becomes a GameResult; an
        internal engine failure becomes TerminationReason.ENGINE_ERROR (result.valid is False).
        Raises: EngineStateError: status was not READY.
        '''
        if self._status is not GameStatus.READY:
            raise EngineStateError('run() requires status READY')
        self._status = GameStatus.RUNNING
        self._t0 = time.monotonic()
        agent_name: str | None = None
        opened = False
        report: GatewayReport | None = None
        self._render_begin()
        try:
            try:
                hello = self._gateway.open()
                opened = True
                agent_name = sanitize_text(hello.agent_name, max_chars=_MAX_AGENT_NAME_CHARS)
                self._gateway.begin_session(self.build_session_init())
                self._render(None)
                self._play()
            except AgentFault as fault:
                self._terminate_fault(fault)
            except Exception as exc:
                message = sanitize_text(f'{type(exc).__name__}: {exc}', max_chars=_MAX_MESSAGE_CHARS)
                self._terminate(TerminationReason.ENGINE_ERROR, message)
        finally:
            report = self._shutdown(opened)
        result = self._result(agent_name, report)
        try:
            self._renderer.end(result)
        except Exception:
            self._renderer = NullRenderer()  # invariant E5: a renderer failure never affects the game
        return result

    def _play(self) -> None:
        '''Turn protocol of spec 4.2.3, steps 4 to 14.'''
        deadline = self._t0 + self._limits.game_wall_clock_s
        observation = self.build_observation()
        while True:
            reply = self._gateway.request_action(
                observation, timeout_s=self._timeout_left(deadline)
            )
            while (problem := self._reject_reason(reply, observation.step_index)) is not None:
                self._invalid_strikes += 1
                if self._invalid_strikes > self._limits.max_invalid_actions:  # spec 4.2.3: 'more than'
                    self._terminate(
                        TerminationReason.INVALID_ACTION_LIMIT, 'too many invalid replies'
                    )
                    return
                notice = InvalidActionDTO(
                    step_index=observation.step_index,
                    reason=problem,
                    strikes_used=self._invalid_strikes,
                    strikes_allowed=self._limits.max_invalid_actions,
                )
                reply = self._gateway.request_replacement_action(
                    notice, timeout_s=self._timeout_left(deadline)
                )
            step = self.apply_action(Action.parse(reply.action))
            self._render(step)
            if step.terminal:
                return
            observation = self.build_observation()

    @staticmethod
    def _reject_reason(reply: ActionDTO, step_index: int) -> str | None:
        '''Post: None if the reply is acceptable, else the InvalidActionDTO reason code.'''
        if reply.step_index != step_index:
            return 'STEP_MISMATCH'
        try:
            Action.parse(reply.action)
        except InvalidActionError:
            return 'UNKNOWN_ACTION'
        return None

    def _timeout_left(self, deadline: float) -> float:
        '''Raises: AgentTimeoutError: the game wall clock ran out.'''
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise AgentTimeoutError('the game wall clock ran out')
        return min(self._limits.action_timeout_s, remaining)

    @staticmethod
    def _reason_for(fault: AgentFault) -> TerminationReason:
        '''Post: the fault-mapping table of spec 4.9.'''
        if isinstance(fault, AgentLaunchError):
            if fault.infrastructure:
                return TerminationReason.ENGINE_ERROR
            return TerminationReason.AGENT_ERROR
        if isinstance(fault, AgentTimeoutError):
            return TerminationReason.AGENT_TIMEOUT
        if isinstance(fault, AgentCrashedError):
            return TerminationReason.AGENT_CRASH
        if isinstance(fault, AgentReportedError):
            return TerminationReason.AGENT_ERROR
        if isinstance(fault, ProtocolViolationError):
            return TerminationReason.PROTOCOL_VIOLATION
        return TerminationReason.AGENT_ERROR

    def _terminate_fault(self, fault: AgentFault) -> None:
        '''Post: the game ends with the reason mapped from `fault`.'''
        if isinstance(fault, AgentReportedError):
            text = f'{fault.dto.error_type}: {fault.dto.message}'
        else:
            text = f'{type(fault).__name__}: {fault}'
        self._terminate(self._reason_for(fault), sanitize_text(text, max_chars=_MAX_MESSAGE_CHARS))

    def _terminate(self, reason: TerminationReason, message: str | None) -> None:
        '''Post: records FORFEIT where spec 4.2.2 says so, then finishes. No-op if already finished.'''
        if self._status is GameStatus.FINISHED:
            return
        if reason in _FORFEIT_REASONS:
            self._score.record(ScoreEvent.FORFEIT, step_index=self._agent.actions_taken)
        self._finish(reason, message)

    def _finish(self, reason: TerminationReason, fault_message: str | None) -> None:
        '''Post: score finalized, log closed, status FINISHED. Idempotent.'''
        if self._status is GameStatus.FINISHED:
            return
        final = self._score.finalize()
        self._termination = reason
        self._final = final
        self._fault_message = fault_message
        self._wall_time_s = time.monotonic() - self._t0
        self._status = GameStatus.FINISHED
        footer = LogFooter(
            termination=reason,
            outcome=reason.outcome,
            final_score=final.total,
            actions_taken=self._agent.actions_taken,
            fault_message=fault_message,
            wall_time_s=self._wall_time_s,
        )
        try:
            self._log.close(footer)
        except Exception:
            pass  # invariant E5: a log failure never changes score, status or agent-facing messages

    def _log_append(self, entry: ActionLogEntry) -> None:
        '''Post: entry appended; a log failure is ignored (invariant E5).'''
        try:
            self._log.append(entry)
        except Exception:
            pass

    def _shutdown(self, opened: bool) -> GatewayReport | None:
        '''Post: best-effort GameOverDTO, then gateway.close() exactly once.'''
        termination, final = self._termination, self._final
        if opened and termination is not None and final is not None:
            try:
                self._gateway.end_session(
                    GameOverDTO(
                        outcome=termination.outcome,
                        final_score=final.total,
                        actions_taken=self._agent.actions_taken,
                    )
                )
            except Exception:
                pass  # the agent may already be dead
        try:
            return self._gateway.close()
        except Exception:
            return None

    def _result(self, agent_name: str | None, report: GatewayReport | None) -> GameResult:
        '''Raises: EngineStateError: the game did not finish.'''
        termination, final = self._termination, self._final
        if termination is None or final is None:
            raise EngineStateError('the game did not finish')
        return GameResult(
            run=self._run_info,
            termination=termination,
            final_score=final.total,
            score_breakdown=final.breakdown,
            actions_taken=self._agent.actions_taken,
            arrows_used=self._initial_arrows - self._agent.arrows,
            wumpuses_killed=self._initial_arrows - self._world.wumpuses_alive,
            gold_grabbed=self._agent.has_gold,
            wall_time_s=self._wall_time_s,
            agent_name=agent_name,
            fault_message=self._fault_message,
            gateway=report,
        )

    def _frame(self, step: StepResult | None) -> RenderFrame:
        '''Post: an OBSERVER frame; `step` is None for the initial frame before the first action.'''
        termination = None if step is None else step.termination
        return RenderFrame(
            mode=_RENDER_MODE,
            turn=self._agent.actions_taken,
            agent=self._agent.snapshot(),
            last_action=None if step is None else step.action,
            last_percept=self._percept if step is None else step.percept,
            score_total=self._score.total,
            score_delta=0 if step is None else step.score_delta,
            terminal=False if step is None else step.terminal,
            outcome=None if termination is None else termination.outcome,
            termination=termination,
            world=self._world.snapshot(),
        )

    def _render_begin(self) -> None:
        '''Post: the renderer is started; on failure it is replaced by a NullRenderer (invariant E5).'''
        run = self._run_info
        try:
            self._renderer.begin(f'{run.agent_id} on {run.scenario_id}', _RENDER_MODE)
        except Exception:
            self._renderer = NullRenderer()

    def _render(self, step: StepResult | None) -> None:
        '''Post: one frame shown; on failure the renderer is replaced by a NullRenderer (invariant E5).'''
        try:
            self._renderer.render(self._frame(step))
        except Exception:
            self._renderer = NullRenderer()