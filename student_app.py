# student_app.py
'''Wumpus World - Student Edition: write an agent, then run or step it on the pygame board.

Run it from the project folder:  python student_app.py   (needs `pip install pygame`; tkinter ships with Python)

  Run All  plays the game to the end at high speed.
  Step     plays exactly one action (during Run All: pauses it).
  Reset    puts the board back to turn 0 without touching the code box. Pressed on a board that is already
           at turn 0 it draws a new random map.

How your code is run (the same way the instructor's Evaluator runs it):

  code box -> temporary .py file -> SubprocessLauncher -> `python -s -P -B -m wumpus_sdk.host` (child process)
           <-> AgentGateway (JSON frames over stdin / stdout) <-> GameEngine -> PygameRenderer

  Your agent lives in a separate Python process. It only ever receives the observation the rules allow
  (percepts, score rules, seed) and sends back an action, so it cannot reach the World object that lives in
  this process (no gc, inspect or sys._getframe tricks) and cannot see the true map. A slow act() (more than
  the action timeout) or a crash forfeits the game, exactly as in the tournament.

Threads: GameEngine.run() plays a whole game in one blocking call, so it runs on a worker thread. The
worker never touches Tk or pygame: its renderer only puts frames on a queue, and the Tk main thread draws
them with PygameRenderer. A gateway wrapper makes the worker wait for a "turn" from the Step / Run All
buttons before it asks the child process for the next action, so the window never freezes.

print() note: the child's stdout carries the protocol. Debug with  print(..., file=sys.stderr)  instead.
'''
from __future__ import annotations

import ast
import os
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk
from typing import Final

if sys.version_info < (3, 11):  # the framework uses enum.StrEnum and typing.assert_never
    raise SystemExit('The Wumpus World framework needs Python 3.11 or newer.')

_ROOT: Final[Path] = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:  # the framework packages resolve from here
    sys.path.insert(0, str(_ROOT))

from wumpus_engine.config.schema import (  # noqa: E402
    EngineLimits,
    SandboxLevel,
    ScenarioConfig,
    ScenarioVisibility,
)
from wumpus_engine.core.agent_state import AgentState  # noqa: E402
from wumpus_engine.core.errors import AgentReportedError, RendererError, WumpusEngineError  # noqa: E402
from wumpus_engine.core.game_engine import GameEngine  # noqa: E402
from wumpus_engine.core.results import GameResult, GatewayReport, RunInfo, TerminationReason  # noqa: E402
from wumpus_engine.core.rules import DEFAULT_SCORE_TABLE, SolvabilityPolicy  # noqa: E402
from wumpus_engine.core.world import World  # noqa: E402
from wumpus_engine.core.world_generator import WorldGenerator  # noqa: E402
from wumpus_engine.display.pygame_renderer import PygameRenderer  # noqa: E402
from wumpus_engine.display.renderer import RenderFrame, RenderMode, Renderer  # noqa: E402
from wumpus_engine.gateway.agent_gateway import AgentGateway, AgentPort  # noqa: E402
from wumpus_engine.gateway.subprocess_launcher import SubprocessLauncher  # noqa: E402
from wumpus_sdk.dto import (  # noqa: E402
    ActionDTO,
    AgentErrorDTO,
    GameOverDTO,
    HelloDTO,
    InvalidActionDTO,
    ObservationDTO,
    SessionInitDTO,
)
from wumpus_sdk.types import START_POSITION  # noqa: E402

# ---- settings ------------------------------------------------------------------------------------------
AGENT_CLASS: Final[str] = 'StudentAgent'
PIT_RANGE: Final[tuple[int, int]] = (0, 20)
WUMPUS_RANGE: Final[tuple[int, int]] = (1, 4)
DEFAULT_PITS: Final[int] = 2
DEFAULT_WUMPUSES: Final[int] = 1
MAX_ACTIONS: Final[int] = 500  # reaching it is a forfeit (ACTION_LIMIT)
RUN_DELAY_MS: Final[int] = 20  # Run All: time per action (about 50 actions per second)
TICK_MS: Final[int] = 8  # while a game is moving: how often the window looks for the engine's next frame
IDLE_TICK_MS: Final[int] = 30  # otherwise: how often it services the pygame window
SLOW_AGENT_S: Final[float] = 2.0  # after this long without an answer the status line says so
# A stepped game may stay open for hours, so the engine's wall clock (300 s by default) must not end it.
# The per-action timeout (EngineLimits.action_timeout_s) stays at its default: it is the tournament's rule.
LIMITS: Final[EngineLimits] = EngineLimits(max_actions=MAX_ACTIONS, game_wall_clock_s=7 * 24 * 3600.0)

_AGENT_FAULTS: Final[frozenset[TerminationReason]] = frozenset({
    TerminationReason.AGENT_ERROR,
    TerminationReason.AGENT_CRASH,
    TerminationReason.AGENT_TIMEOUT,
    TerminationReason.PROTOCOL_VIOLATION,
    TerminationReason.INVALID_ACTION_LIMIT,
    TerminationReason.ENGINE_ERROR,
})

TEMPLATE: Final[str] = '''\
# Write your agent here. The class must be called StudentAgent and derive from Agent.
from wumpus_sdk.actions import Action
from wumpus_sdk.agent import Agent
from wumpus_sdk.dto import ObservationDTO


class StudentAgent(Agent):
    """Example: walks straight ahead, turns left at a wall and grabs the gold if it stands on it."""

    def act(self, observation: ObservationDTO) -> Action:
        percept = observation.percept
        if percept.glitter:
            return Action.GRAB
        if percept.bump:
            return Action.TURN_LEFT
        return Action.MOVE_FORWARD
'''

_AGENT_EXAMPLE: Final[str] = (
    '    from wumpus_sdk.agent import Agent\n\n    class StudentAgent(Agent):\n'
    '        def act(self, observation):\n            ...\n')


# ---- problems shown in the popup -----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Problem:
    '''Something wrong with the student's code, ready to be shown in the error popup.'''

    summary: str  # one line
    details: str = ''  # the syntax error excerpt or the traceback, for a monospace box
    line: int | None = None  # the line of the code box to highlight


class AgentLoadError(Exception):
    '''The code in the box cannot be started as an agent.'''

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem.summary)
        self.problem = problem


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    return base.attr if isinstance(base, ast.Attribute) else ''


def check_source(source: str) -> str:
    '''Post: the name of the agent class: StudentAgent or, without one, the only class derived from Agent.
    The code is only parsed and compiled here, never executed: it runs solely in the child process.
    Raises: AgentLoadError: empty box, syntax error (with the line and the caret excerpt) or no agent class.
    '''
    if not source.strip():
        raise AgentLoadError(Problem('The code box is empty', f'Write your agent first:\n\n{_AGENT_EXAMPLE}'))
    try:
        tree = ast.parse(source, filename='your code')
        compile(tree, 'your code', 'exec')  # also finds what only the compiler reports, e.g. 'return' outside a function
    except SyntaxError as exc:
        raise AgentLoadError(Problem(
            f'Syntax error on line {exc.lineno}: {exc.msg}',
            ''.join(traceback.format_exception_only(exc)), exc.lineno)) from exc
    except (ValueError, RecursionError, MemoryError, OverflowError) as exc:  # null bytes, absurd nesting
        raise AgentLoadError(Problem('Your code cannot be compiled', f'{type(exc).__name__}: {exc}\n')) from exc
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if any(node.name == AGENT_CLASS for node in classes):
        return AGENT_CLASS
    derived = [node.name for node in classes if any(_base_name(base) == 'Agent' for base in node.bases)]
    if len(derived) == 1:
        return derived[0]
    found = f'Agent classes found: {", ".join(derived)}\nName the one to run {AGENT_CLASS}.\n\n' if derived else ''
    raise AgentLoadError(Problem(f'No class {AGENT_CLASS} found', f'{found}Your code must define:\n\n{_AGENT_EXAMPLE}'))


_FRAME_RE: Final[re.Pattern[str]] = re.compile(r'File "([^"]+)", line (\d+)')
_NAME_HINTS: Final[dict[str, str]] = {
    'Agent': 'from wumpus_sdk.agent import Agent',
    'Action': 'from wumpus_sdk.actions import Action',
    'ObservationDTO': 'from wumpus_sdk.dto import ObservationDTO',
    'Percept': 'from wumpus_sdk.percept import Percept',
    'Direction': 'from wumpus_sdk.types import Direction',
    'Position': 'from wumpus_sdk.types import Position',
}
_PRINT_HINT: Final[str] = (
    '\nHint: the agent process talks to the game over its stdout, so print() breaks the game.\n'
    'Debug with:  import sys;  print(..., file=sys.stderr)\n')


def tidy_traceback(text: str, module_file: Path) -> tuple[str, int | None]:
    '''Post: a traceback written by the child process without its internal frames (wumpus_sdk/host.py),
    with the temporary file shown as 'your code', and the last line number inside the student's file.
    '''
    shown_path = re.compile(r'File "[^"]*' + re.escape(module_file.name) + '"')  # 8.3 and long path forms differ
    line: int | None = None
    kept: list[str] = []
    skipping = False
    for raw in text.splitlines():
        match = _FRAME_RE.search(raw)
        if match is not None:
            skipping = Path(match.group(1)).name == 'host.py'
            if not skipping and Path(match.group(1)).name == module_file.name:
                line = int(match.group(2))
        elif skipping and raw.startswith('    '):
            continue  # the source line / carets of a skipped frame
        else:
            skipping = False
        if not skipping:
            kept.append(shown_path.sub('File "your code"', raw))
    return '\n'.join(kept) + '\n', line


# ---- one game: the engine on a worker thread -------------------------------------------------------------
class _Cancelled(Exception):
    '''The window abandoned this game (Reset, a new game, or it was closed).'''


@dataclass(frozen=True, slots=True)
class _EngineCrash:
    '''GameEngine.run() itself raised; it never should.'''

    details: str


_Message = tuple[str, RenderMode] | RenderFrame | GameResult | _EngineCrash  # begin, render, end, crash


class _Session:
    '''What the Tk thread and the engine thread of one game share. Every method is thread-safe.'''

    def __init__(self, source: str, module_file: Path) -> None:
        self.source = source  # the text the agent was built from
        self.module_file = module_file  # the temporary file the child process imports
        self.messages: queue.SimpleQueue[_Message] = queue.SimpleQueue()  # engine thread -> Tk thread
        self.reported: AgentErrorDTO | None = None  # the exception the child process reported, if any
        self.stderr = ''  # what the child process wrote to stderr (read when the gateway closes)
        self.thread: threading.Thread | None = None
        self._turns = threading.Condition()
        self._granted = 0
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def grant_turn(self) -> None:
        '''Tk thread. Post: the agent may take one more action.'''
        with self._turns:
            self._granted += 1
            self._turns.notify_all()

    def cancel(self) -> None:
        '''Tk thread. Post: the engine thread ends its game at the next action it asks for.'''
        with self._turns:
            self._cancelled = True
            self._turns.notify_all()

    def wait_for_turn(self) -> None:
        '''Engine thread. Post: returns once a granted turn was consumed. Raises: _Cancelled.'''
        with self._turns:
            while self._granted == 0 and not self._cancelled:
                self._turns.wait()
            if self._cancelled:
                raise _Cancelled('the window abandoned this game')
            self._granted -= 1


class _QueueRenderer(Renderer):
    '''The engine's renderer. It draws nothing: it hands every call to the Tk thread through the session's
    queue (a RenderFrame is immutable, so it can cross threads as it is).
    '''

    def __init__(self, session: _Session) -> None:
        self._session = session

    def begin(self, title: str, mode: RenderMode) -> None:
        self._session.messages.put((title, mode))

    def render(self, frame: RenderFrame) -> None:
        self._session.messages.put(frame)

    def end(self, result: GameResult) -> None:
        self._session.messages.put(result)


class _SteppedGateway(AgentPort):
    '''The AgentPort given to GameEngine: the real AgentGateway (child process, JSON frames, timeouts), plus
    a wait for a granted turn before every action. It also records what the child reported or wrote to
    stderr, for the error popup (the engine's GameResult only keeps a one-line message).
    '''

    def __init__(self, inner: AgentGateway, launcher: SubprocessLauncher, session: _Session) -> None:
        self._inner = inner
        self._launcher = launcher
        self._session = session
        self._observation: ObservationDTO | None = None

    @property
    def isolation_level(self) -> SandboxLevel:
        return self._inner.isolation_level

    def _record(self, exc: AgentReportedError) -> None:
        if self._session.reported is None:
            self._session.reported = exc.dto

    def open(self) -> HelloDTO:
        try:
            return self._inner.open()
        except AgentReportedError as exc:
            self._record(exc)
            raise

    def begin_session(self, init: SessionInitDTO) -> None:
        self._inner.begin_session(init)

    def request_action(self, observation: ObservationDTO, *, timeout_s: float) -> ActionDTO:
        self._session.wait_for_turn()  # the child is not asked, and no timeout runs, until the window says go
        self._observation = observation
        try:
            return self._inner.request_action(observation, timeout_s=timeout_s)
        except AgentReportedError as exc:
            self._record(exc)
            raise

    def request_replacement_action(self, notice: InvalidActionDTO, *, timeout_s: float) -> ActionDTO:
        if self._session.cancelled:  # same turn as the action just asked: no new grant is needed
            raise _Cancelled('the window abandoned this game')
        try:
            return self._inner.request_replacement_action(notice, timeout_s=timeout_s)
        except AgentReportedError as exc:
            self._record(exc)
            raise

    def end_session(self, result: GameOverDTO) -> None:
        self._inner.end_session(result)

    def close(self) -> GatewayReport:
        report = self._inner.close()
        time.sleep(0.05)  # let the stderr reader thread collect the last lines of a dead child
        self._session.stderr = self._launcher.get_stderr_tail(8192)
        return report


def _run_engine(engine: GameEngine, session: _Session) -> None:
    '''Engine thread: play one game. Its frames and its result reach the window through the queue.'''
    try:
        engine.run()
    except BaseException:  # run() reports every failure as a GameResult; this is the last resort
        session.messages.put(_EngineCrash(traceback.format_exc()))


def fault_problem(session: _Session, result: GameResult) -> Problem | None:
    '''Post: the popup for a game that ended because of the agent (an exception, a crash, a timeout, a broken
    protocol), or None for a game that ended by the rules.
    '''
    reason = result.termination
    if reason not in _AGENT_FAULTS:
        return None
    reported = session.reported
    if reported is not None:
        text, line = tidy_traceback(reported.traceback, session.module_file)
        where = 'while it was loading' if reported.step_index < 0 else f'on turn {reported.step_index + 1}'
        summary = f'Your agent raised {reported.error_type} {where}'
        if line is not None:
            summary += f' (line {line})'
        details = f'{text}\n{reported.error_type}: {reported.message}\n' if not text.rstrip().endswith(reported.message) else text
        if reported.error_type == 'NameError':
            for name, statement in _NAME_HINTS.items():
                if f"'{name}'" in reported.message:
                    details += f'\nHint: add this line at the top of your code:\n    {statement}\n'
                    break
        return Problem(summary, details, line)
    stderr, line = tidy_traceback(session.stderr, session.module_file)
    stderr = stderr.strip()
    headlines = {
        TerminationReason.AGENT_TIMEOUT: 'Your agent was too slow: act() must answer within the action timeout',
        TerminationReason.AGENT_CRASH: 'Your agent process stopped unexpectedly',
        TerminationReason.PROTOCOL_VIOLATION: 'Your agent sent something that is not an action',
        TerminationReason.INVALID_ACTION_LIMIT: 'Your agent returned too many invalid actions',
        TerminationReason.AGENT_ERROR: 'Your agent could not be started',
        TerminationReason.ENGINE_ERROR: 'The game engine failed',
    }
    details = f'{reason.value}: {result.fault_message or "no message"}\n'
    if stderr:
        details += f'\nOutput of your agent on stderr:\n{stderr}\n'
    if reason in (TerminationReason.PROTOCOL_VIOLATION, TerminationReason.AGENT_CRASH):
        details += _PRINT_HINT
    return Problem(headlines[reason], details, line)


# ---- widgets ------------------------------------------------------------------------------------------
class _Phase(Enum):
    IDLE = 'idle'  # a fresh board at turn 0 (or no board at all); no game yet
    PAUSED = 'paused'  # a game is in progress and waits for Step or Run All
    RUNNING = 'running'  # Run All: one action after the other
    OVER = 'over'  # the game ended; the final board stays on screen


def _random_seed() -> int:
    return random.SystemRandom().randrange(1_000_000)  # short enough to read and quote


def _make_code_box(parent: tk.Misc, font: tkfont.Font, *, width: int, height: int) -> tuple[ttk.Frame, tk.Text]:
    '''Post: a frame holding a monospace Text for Python code with both scrollbars. Tab inserts four spaces,
    Ctrl+A selects everything, and the 'error' tag (a red line) goes away at the next key.
    '''
    box = ttk.Frame(parent)
    box.columnconfigure(0, weight=1)
    box.rowconfigure(0, weight=1)
    code = tk.Text(box, width=width, height=height, wrap='none', undo=True, font=font, padx=6, pady=6)
    vertical = ttk.Scrollbar(box, orient='vertical', command=code.yview)
    horizontal = ttk.Scrollbar(box, orient='horizontal', command=code.xview)
    code.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    code.grid(row=0, column=0, sticky='nsew')
    vertical.grid(row=0, column=1, sticky='ns')
    horizontal.grid(row=1, column=0, sticky='ew')
    code.tag_configure('error', background='#ffd0d0', foreground='#000000')

    def indent(_event: tk.Event[tk.Text]) -> str:
        code.insert('insert', '    ')
        return 'break'

    def select_all(_event: tk.Event[tk.Text]) -> str:
        code.tag_add('sel', '1.0', 'end-1c')
        return 'break'

    code.bind('<Tab>', indent)
    code.bind('<Control-a>', select_all)
    code.bind('<Control-A>', select_all)
    code.bind('<Key>', lambda _event: code.tag_remove('error', '1.0', 'end'), add='+')
    return box, code


class ProblemPopup:
    '''The error popup: a one-line summary and, in a monospace box, the details (the excerpt of a syntax
    error with its caret, or a traceback). Only one is open at a time.
    '''

    def __init__(self, root: tk.Tk, mono: tkfont.Font) -> None:
        self._root = root
        self._mono = mono
        self._bold = tkfont.nametofont('TkDefaultFont').copy()  # kept: a Tk font disappears with its object
        self._bold.configure(weight='bold')
        self.window: tk.Toplevel | None = None

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def show(self, problem: Problem, title: str, code: tk.Text | None = None) -> None:
        '''Post: the popup is open. With `code`, the faulty line of that code box is highlighted.'''
        if code is not None:
            code.tag_remove('error', '1.0', 'end')
            if problem.line is not None:
                code.tag_add('error', f'{problem.line}.0', f'{problem.line}.0 lineend +1c')
                code.see(f'{problem.line}.0')
        self.close()
        popup = tk.Toplevel(self._root)
        self.window = popup
        popup.title(title)
        popup.transient(self._root)
        frame = ttk.Frame(popup, padding=12)
        frame.grid(row=0, column=0, sticky='nsew')
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text=problem.summary, font=self._bold, wraplength=640, justify='left').grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
        if problem.details:
            lines = problem.details.rstrip('\n').split('\n')
            text = tk.Text(frame, width=min(max(max(len(line) for line in lines) + 2, 50), 100),
                           height=min(max(len(lines), 3), 22), wrap='none', font=self._mono, padx=6, pady=6)
            scroll = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            text.insert('1.0', '\n'.join(lines))
            text.configure(state='disabled')
            text.see('end')  # the last lines of a traceback say what went wrong
            text.grid(row=1, column=0, sticky='nsew')
            scroll.grid(row=1, column=1, sticky='ns')
        button = ttk.Button(frame, text='Close', command=self.close)
        button.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        popup.protocol('WM_DELETE_WINDOW', self.close)
        popup.bind('<Escape>', lambda _event: self.close())
        popup.bind('<Return>', lambda _event: self.close())
        popup.update_idletasks()  # beside the window when there is room, so the highlighted line stays visible
        left = self._root.winfo_rootx() + self._root.winfo_width() + 12
        if left + popup.winfo_reqwidth() > self._root.winfo_screenwidth():
            left = self._root.winfo_rootx() + 40
        popup.geometry(f'+{left}+{self._root.winfo_rooty() + 60}')
        popup.lift()
        button.focus_set()
        self._root.bell()


# ---- the application -----------------------------------------------------------------------------------
class StudentApp:
    '''The Student Edition window: code box, difficulty spinboxes, Step / Run All / Reset, and the pygame
    board. Everything here runs on the Tk main thread, except GameEngine.run() (worker thread per game).
    '''

    def __init__(self, root: tk.Tk) -> None:
        self._root = root
        root.title('Wumpus World - Student Edition')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self._mono = tkfont.nametofont('TkFixedFont').copy()  # kept: a Tk font disappears with its object
        self._mono.configure(size=10)
        self._popup = ProblemPopup(root, self._mono)
        self._directory = Path(tempfile.mkdtemp(prefix='wumpus_student_'))  # the child's source_dir
        self._serial = 0
        self._session: _Session | None = None
        self._phase = _Phase.IDLE
        self._seed = _random_seed()  # the map; kept until Reset draws a new one
        self._agent_seed = _random_seed()  # the agent's self.rng; kept with the map so a replay is identical
        self._shown: tuple[int, int] | None = None  # (pits, wumpuses) of the board on screen
        self._in_flight = 0  # turns granted whose frame has not arrived yet
        self._last_progress = time.monotonic()
        self._granted_at = 0.0  # time.monotonic() of the latest granted turn, for the pace of Run All
        self._grant_job: str | None = None
        self._tick_job: str | None = None
        self._closing = False
        self._status = tk.StringVar(value='Starting...')
        self._pits = tk.StringVar(value=str(DEFAULT_PITS))
        self._wumpuses = tk.StringVar(value=str(DEFAULT_WUMPUSES))
        self._code = self._build_widgets()
        cell = self._board_cell_size()
        self._renderer = PygameRenderer(cell_size=cell, step_delay_s=0.0, hold_on_end=False)
        root.update_idletasks()
        self._place_windows(cell)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self._first_board)
        self._tick()

    # ---- widgets -----------------------------------------------------------------------------------------
    def _build_widgets(self) -> tk.Text:
        outer = ttk.Frame(self._root, padding=10)
        outer.grid(row=0, column=0, sticky='nsew')
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text=f'Agent code  -  write your  class {AGENT_CLASS}(Agent): ...  here').grid(
            row=0, column=0, sticky='w', pady=(0, 4))

        box, code = _make_code_box(outer, self._mono, width=76, height=26)
        box.grid(row=1, column=0, sticky='nsew')
        code.insert('1.0', TEMPLATE)
        code.edit_reset()  # the template is not an undo step

        settings = ttk.Frame(outer)
        settings.grid(row=2, column=0, sticky='w', pady=(10, 0))
        digits = (self._root.register(lambda text: text == '' or (text.isdigit() and len(text) <= 2)), '%P')
        ttk.Label(settings, text='Number of Pits').grid(row=0, column=0, sticky='w')
        self._pit_box = ttk.Spinbox(
            settings, from_=PIT_RANGE[0], to=PIT_RANGE[1], width=4, textvariable=self._pits,
            validate='key', validatecommand=digits, command=self._settings_changed)
        self._pit_box.grid(row=0, column=1, padx=(6, 24))
        ttk.Label(settings, text='Number of Wumpuses').grid(row=0, column=2, sticky='w')
        self._wumpus_box = ttk.Spinbox(
            settings, from_=WUMPUS_RANGE[0], to=WUMPUS_RANGE[1], width=4, textvariable=self._wumpuses,
            validate='key', validatecommand=digits, command=self._settings_changed)
        self._wumpus_box.grid(row=0, column=3, padx=(6, 0))
        for spinbox in (self._pit_box, self._wumpus_box):
            spinbox.bind('<Return>', lambda _event: self._settings_changed())
            spinbox.bind('<FocusOut>', lambda _event: self._settings_changed())

        buttons = ttk.Frame(outer)
        buttons.grid(row=3, column=0, sticky='w', pady=(10, 0))
        ttk.Button(buttons, text='Run All', width=12, command=self.run_all).grid(row=0, column=0)
        ttk.Button(buttons, text='Step', width=12, command=self.step).grid(row=0, column=1, padx=8)
        ttk.Button(buttons, text='Reset', width=12, command=self.reset).grid(row=0, column=2)

        ttk.Separator(outer).grid(row=4, column=0, sticky='ew', pady=(10, 6))
        status = ttk.Label(outer, textvariable=self._status, wraplength=560, justify='left')
        status.grid(row=5, column=0, sticky='ew')
        outer.bind('<Configure>', lambda event: status.configure(wraplength=max(event.width - 24, 200)))
        return code

    def _board_cell_size(self) -> int:
        '''Post: the cell size (24 to 34 pixels) that makes the pygame board fit the height of the screen.'''
        screen_h = self._root.winfo_screenheight()
        return max(24, min(34, (screen_h - 46 - 110) // 20))  # 46: the board's margins; 110: title bars, taskbar

    def _place_windows(self, cell: int) -> None:
        '''Post: this window sits at the top left and the pygame board is told to open to its right.'''
        root = self._root
        board_w = 440 + 20 * cell  # margins and side panel of PygameRenderer's layout
        left = 20 + root.winfo_reqwidth() + 24
        if left + board_w > root.winfo_screenwidth():  # no room side by side: push the board to the right edge
            left = max(0, root.winfo_screenwidth() - board_w - 8)
        root.geometry('+20+20')
        os.environ.setdefault('SDL_VIDEO_WINDOW_POS', f'{left},48')  # read by SDL when the window is created

    def _show_problem(self, problem: Problem, title: str) -> None:
        '''Post: the error popup is open and the faulty line of the code box is highlighted.'''
        self._popup.show(problem, title, self._code)

    # ---- settings ----------------------------------------------------------------------------------------
    @staticmethod
    def _read_count(variable: tk.StringVar, bounds: tuple[int, int], default: int) -> int:
        '''Post: the spinbox value as an int inside `bounds`; the box is corrected to show it.'''
        try:
            value = int(variable.get().strip())
        except ValueError:
            value = default
        value = max(bounds[0], min(bounds[1], value))
        if variable.get() != str(value):
            variable.set(str(value))
        return value

    def _settings(self) -> tuple[int, int]:
        return (self._read_count(self._pits, PIT_RANGE, DEFAULT_PITS),
                self._read_count(self._wumpuses, WUMPUS_RANGE, DEFAULT_WUMPUSES))

    @staticmethod
    def _scenario(pits: int, wumpuses: int) -> ScenarioConfig:
        text = (f'{pits} pit{"" if pits == 1 else "s"}, {wumpuses} Wumpus{"" if wumpuses == 1 else "es"}')
        return ScenarioConfig(
            scenario_id=text, difficulty_rank=1, visibility=ScenarioVisibility.PRACTICE,
            pit_count=pits, wumpus_count=wumpuses, solvability=SolvabilityPolicy.REACHABLE_WITH_SHOOTING)

    def _settings_changed(self) -> None:
        '''A spinbox changed: with no game in progress the board is redrawn for the new difficulty at once;
        a game in progress keeps its map until Reset.
        '''
        settings = self._settings()
        if settings == self._shown:
            return
        if self._session is not None:
            self._status.set('The new difficulty applies to the next game: press Reset.')
        elif self._show_board():
            self._status.set(f'{self._map_label()}. Press Step or Run All.')

    def _map_label(self) -> str:
        pits, wumpuses = self._shown if self._shown is not None else self._settings()
        return f'Map {self._seed}: {self._scenario(pits, wumpuses).scenario_id}'

    # ---- the board ---------------------------------------------------------------------------------------
    def _first_board(self) -> None:
        if self._show_board():
            self._status.set(f'{self._map_label()}. Write your agent, then press Step or Run All.')

    def _show_board(self) -> bool:
        '''Post: any game in progress is abandoned and the board shows turn 0 of the map given by the seed
        and the spinboxes (the pygame window is opened if needed). Returns False, after a popup, when the
        map or the window cannot be made.
        '''
        self._abandon_game()
        self._phase = _Phase.IDLE
        pits, wumpuses = self._settings()
        self._shown = (pits, wumpuses)
        scenario = self._scenario(pits, wumpuses)
        try:
            world = World(WorldGenerator().generate(scenario, self._seed))
            frame = RenderFrame(
                mode=RenderMode.OBSERVER, turn=0, agent=AgentState(arrows=wumpuses).snapshot(), last_action=None,
                last_percept=world.perceive(START_POSITION), score_total=0, score_delta=0, terminal=False,
                outcome=None, termination=None, world=world.snapshot(),
                note='New game: press Step or Run All.')
            self._renderer.begin(f'New game on {scenario.scenario_id}', RenderMode.OBSERVER)
            self._renderer.render(frame)
        except WumpusEngineError as exc:  # WorldGenerationError, RendererError
            self._renderer.close()
            self._status.set(f'No board: {exc}')
            self._show_problem(Problem('The board cannot be shown', f'{type(exc).__name__}: {exc}\n'), 'Board error')
            return False
        return True

    def _board_closed(self, failure: str | None = None) -> None:
        '''The user closed the pygame window (X button, ESC or Q), or drawing on it failed (`failure`).'''
        playing = self._session is not None
        self._abandon_game()
        self._renderer.close()
        self._phase = _Phase.IDLE
        what = 'Board window closed.' if failure is None else f'The board window failed ({failure}).'
        ended = ' The game was stopped.' if playing else ''
        self._status.set(f'{what}{ended} Reset, Step or Run All opens it again.')

    # ---- games -------------------------------------------------------------------------------------------
    def _abandon_game(self) -> None:
        '''Post: no game is in progress. The engine thread of an abandoned game ends by itself at its next
        action and terminates the child process (GameEngine.run() always closes the gateway).
        '''
        self._cancel_auto_grant()
        if self._session is not None:
            self._session.cancel()
            self._session = None
        self._in_flight = 0

    def _write_agent(self, source: str) -> Path:
        '''Post: `source` saved as a new module in the temporary folder (older ones deleted); its path.
        Raises: AgentLoadError: the file cannot be written.
        '''
        for old in self._directory.glob('student_agent_*.py'):
            try:
                old.unlink()
            except OSError:
                pass  # an old child still has it open (Windows); it is never imported again
        self._serial += 1
        path = self._directory / f'student_agent_{self._serial}.py'
        try:
            path.write_text(source, encoding='utf-8')
        except OSError as exc:
            raise AgentLoadError(Problem('Cannot write the temporary file', f'{path}\n{exc}\n')) from exc
        return path

    def _start_game(self) -> bool:
        '''Post: the code box was checked and saved to a temporary file, and a new game on the current map
        runs on its own thread: GameEngine -> AgentGateway -> SubprocessLauncher -> child process. The game
        waits for its first turn. Returns False, after a popup, if the code cannot be started.
        '''
        self._abandon_game()
        source = self._code.get('1.0', 'end-1c')
        self._code.tag_remove('error', '1.0', 'end')
        try:
            class_name = check_source(source)  # syntax errors are caught here, before any process starts
            module_file = self._write_agent(source)
        except AgentLoadError as exc:
            self._status.set(f'{exc.problem.summary}. Fix the code and press Step or Run All again.')
            self._show_problem(exc.problem, 'Cannot run your agent')
            return False
        pits, wumpuses = self._settings()
        scenario = self._scenario(pits, wumpuses)
        session = _Session(source, module_file)
        launcher = SubprocessLauncher(
            entry_point=f'{module_file.stem}:{class_name}',
            source_dir=self._directory,
            isolation_level=SandboxLevel.PROCESS,
        )
        gateway = _SteppedGateway(AgentGateway(launcher), launcher, session)
        run_info = RunInfo(
            run_id=uuid.uuid4().hex, agent_id=class_name, scenario_id=scenario.scenario_id,
            world_seed=self._seed, agent_seed=self._agent_seed, isolation=SandboxLevel.PROCESS)
        try:
            engine = GameEngine(
                scenario=scenario, run_info=run_info, limits=LIMITS, score_table=DEFAULT_SCORE_TABLE,
                gateway=gateway, renderer=_QueueRenderer(session))
        except WumpusEngineError as exc:
            self._status.set(f'No game: {exc}')
            self._show_problem(Problem('The game cannot be created', f'{type(exc).__name__}: {exc}\n'), 'Game error')
            return False
        self._session = session
        self._shown = (pits, wumpuses)
        self._last_progress = time.monotonic()
        session.thread = threading.Thread(target=_run_engine, args=(engine, session), name='wumpus-engine', daemon=True)
        session.thread.start()
        return True

    def _grant_turn(self) -> None:
        if self._session is None:
            return
        self._granted_at = time.monotonic()
        if self._in_flight == 0:
            self._last_progress = self._granted_at
        self._in_flight += 1
        self._session.grant_turn()

    def _auto_grant(self) -> None:
        self._grant_job = None
        if self._phase is _Phase.RUNNING and self._in_flight == 0:
            self._grant_turn()

    def _cancel_auto_grant(self) -> None:
        if self._grant_job is not None:
            self._root.after_cancel(self._grant_job)
            self._grant_job = None

    def _code_changed_hint(self) -> str:
        session = self._session
        if session is not None and session.source != self._code.get('1.0', 'end-1c'):
            return '  The code box changed: this game still runs the old code, Reset starts over with the new one.'
        return ''

    # ---- buttons -----------------------------------------------------------------------------------------
    def step(self) -> None:
        '''Step: exactly one action. Starts a game (on the map shown) if none is in progress; during
        Run All it pauses the run, and the action already under way is the step.
        '''
        if self._session is None:
            if not self._start_game():
                return
            self._phase = _Phase.PAUSED
        elif self._phase is _Phase.RUNNING:
            self._phase = _Phase.PAUSED
            self._cancel_auto_grant()
            if self._in_flight:
                return
        self._grant_turn()
        self._status.set(f'Step...{self._code_changed_hint()}')

    def run_all(self) -> None:
        '''Run All: play to the end of the game, one action every RUN_DELAY_MS. Starts a game if needed.'''
        if self._session is None:
            if not self._start_game():
                return
        elif self._phase is _Phase.RUNNING:
            return
        self._phase = _Phase.RUNNING
        if self._in_flight == 0:
            self._grant_turn()
        self._status.set(f'Running...  (Step pauses){self._code_changed_hint()}')

    def reset(self) -> None:
        '''Reset: back to turn 0 of the same map; the code box is not touched. On a board that is already
        at turn 0, a new random map instead.
        '''
        untouched = self._phase is _Phase.IDLE and self._session is None and self._renderer.is_open
        if untouched:
            self._seed, self._agent_seed = _random_seed(), _random_seed()
        if not self._show_board():
            return
        if untouched:
            self._status.set(f'New map. {self._map_label()}. Press Step or Run All.')
        else:
            self._status.set(f'Board reset. {self._map_label()} (same map; press Reset again for a new one).')

    # ---- the engine's frames -----------------------------------------------------------------------------
    def _tick(self) -> None:
        '''Every few milliseconds: keep the pygame window responsive, then draw what the engine thread
        has queued. This is the only place where the app touches pygame while a game is on.
        '''
        if self._closing:
            return
        try:
            if self._renderer.is_open and not self._renderer.pump():
                self._board_closed()
            self._drain()
            self._watch_slow_agent()
        finally:
            moving = self._phase is _Phase.RUNNING or self._in_flight > 0
            self._tick_job = self._root.after(TICK_MS if moving else IDLE_TICK_MS, self._tick)

    def _drain(self) -> None:
        session = self._session
        while session is not None and session is self._session:
            try:
                message = session.messages.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(session, message)
            except RendererError as exc:  # the window was closed while a frame was drawn, or pygame failed
                self._board_closed(None if 'closed by the user' in str(exc) else str(exc))

    def _handle(self, session: _Session, message: _Message) -> None:
        if isinstance(message, RenderFrame):
            self._renderer.render(message)
            self._last_progress = time.monotonic()
            if message.last_action is not None and self._in_flight:
                self._in_flight -= 1
            if message.terminal:
                return  # the GameResult follows
            action = 'start' if message.last_action is None else message.last_action.value
            now = f'Turn {message.turn}: {action}, score {message.score_total}.'
            if self._phase is _Phase.RUNNING:
                self._status.set(f'{now}  Running...  (Step pauses)')
                if self._in_flight == 0 and self._grant_job is None:
                    spent_ms = int((time.monotonic() - self._granted_at) * 1000)
                    self._grant_job = self._root.after(max(RUN_DELAY_MS - spent_ms, 1), self._auto_grant)
            elif self._in_flight == 0:
                self._status.set(f'{now}  Paused: Step plays one action, Run All plays to the end.'
                                 f'{self._code_changed_hint()}')
        elif isinstance(message, GameResult):
            self._renderer.end(message)
            self._game_over(session, message)
        elif isinstance(message, _EngineCrash):
            self._abandon_game()
            self._phase = _Phase.OVER
            self._status.set('The game engine failed; see the popup.')
            self._show_problem(Problem('Internal error in the game engine', message.details), 'Engine error')
        else:
            title, mode = message
            self._renderer.begin(title, mode)

    def _game_over(self, session: _Session, result: GameResult) -> None:
        self._cancel_auto_grant()
        self._session = None
        self._in_flight = 0
        self._phase = _Phase.OVER
        self._status.set(
            f'Game over: {result.termination.value}, score {result.final_score} after '
            f'{result.actions_taken} actions.  Step / Run All play this map again with the code in the box; '
            f'Reset shows it at turn 0.')
        problem = fault_problem(session, result)
        if problem is not None:
            self._show_problem(problem, 'Your agent has a problem')

    def _watch_slow_agent(self) -> None:
        if self._session is None or self._in_flight == 0:
            return
        waited = time.monotonic() - self._last_progress
        if waited >= SLOW_AGENT_S:
            self._status.set(f'Waiting for the agent: act() has not answered for {waited:.0f} s. '
                             f'Reset abandons this game.')

    # ---- shutdown ----------------------------------------------------------------------------------------
    def close(self) -> None:
        '''Post: the game is abandoned (its child process is terminated), the board window is closed, the
        temporary files are deleted and the window is destroyed.
        '''
        self._closing = True
        session = self._session
        self._abandon_game()
        if self._tick_job is not None:
            self._root.after_cancel(self._tick_job)
            self._tick_job = None
        if session is not None and session.thread is not None:
            session.thread.join(timeout=3.0)  # the engine thread terminates the child process on its way out
        self._renderer.close()
        self._popup.close()
        shutil.rmtree(self._directory, ignore_errors=True)
        self._root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f'Cannot open the window: {exc}') from exc
    StudentApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
