# wumpus_engine/display/pygame_renderer.py
'''Graphical Observer for the Wumpus World: a pygame window with the 20 x 20 board and a side panel.

The side panel shows the current turn, the percepts, a trace log of the latest turns (action, score and
percepts of each one) and a legend. When the game ends the window stays open, with the final board, the
agent's path and the summary, until the user closes it (the window's X button, ESC or Q).

Percept marks on the grid: breeze is a blue circle, stench a green cross, glitter a yellow dot. They are
drawn in two separate layers. The agent layer (bold) shows what the agent perceived on the cells it stood
on. The true-map layer (thin, translucent) exists in OBSERVER mode only and marks every cell of the board
from the true map: breeze next to a pit, stench next to a living Wumpus, glitter on the untaken gold.

Usage: GameEngine(..., renderer=PygameRenderer()). Needs `pip install pygame`. For a demo on a dummy
world without the engine run `python -m wumpus_engine.display.pygame_renderer` (add --help for options).

A host program with its own event loop (for example the Tkinter control panel in main.py) builds the
renderer with hold_on_end=False, so that end() returns at once and leaves the final board on screen, and
calls pump() regularly to keep the window responsive.
'''
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')  # silence pygame's import banner

try:
    import pygame
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError('PygameRenderer needs the pygame package: pip install pygame') from exc

from wumpus_engine.core.errors import RendererError
from wumpus_engine.core.results import GameResult
from wumpus_engine.core.world import WorldSnapshot
from wumpus_engine.display.renderer import RenderFrame, RenderMode, Renderer
from wumpus_sdk.actions import Action
from wumpus_sdk.dto import Outcome
from wumpus_sdk.percept import Percept
from wumpus_sdk.types import GRID_HEIGHT, GRID_WIDTH, MIN_COORD, Direction, Position

Color = tuple[int, int, int]
Paint = tuple[int, int, int] | tuple[int, int, int, int]  # RGB, or RGBA on a per-pixel-alpha layer
Point = tuple[float, float]

# ---- palette -----------------------------------------------------------------------------------------
_BG: Final[Color] = (20, 22, 28)
_BOARD_A: Final[Color] = (44, 50, 62)
_BOARD_B: Final[Color] = (50, 57, 70)
_VISITED_A: Final[Color] = (58, 92, 78)
_VISITED_B: Final[Color] = (64, 100, 85)
_GRID_LINE: Final[Color] = (28, 32, 42)
_LABEL: Final[Color] = (150, 160, 182)
_PIT_RIM: Final[Color] = (96, 66, 44)
_PIT_HOLE: Final[Color] = (8, 8, 11)
_WUMPUS: Final[Color] = (206, 66, 88)
_WUMPUS_DEAD: Final[Color] = (112, 114, 124)
_GOLD: Final[Color] = (252, 204, 44)
_GOLD_RIM: Final[Color] = (168, 118, 12)
_GOLD_SHINE: Final[Color] = (255, 240, 170)
_GOLD_FADED: Final[Color] = (150, 128, 60)
_AGENT: Final[Color] = (72, 162, 255)
_AGENT_DEAD: Final[Color] = (130, 132, 142)
_TRAIL: Final[Color] = (120, 198, 255)
_SHOT: Final[Color] = (255, 150, 40)
_PANEL: Final[Color] = (32, 35, 46)
_TEXT: Final[Color] = (232, 234, 242)
_DIM: Final[Color] = (128, 136, 154)
_GOOD: Final[Color] = (96, 212, 126)
_BAD: Final[Color] = (242, 94, 94)
_AMBER: Final[Color] = (252, 192, 64)
_WHITE: Final[Color] = (255, 255, 255)
_BLACK: Final[Color] = (0, 0, 0)
# Percept marks on the grid. Agent layer: what the agent perceived on a cell it stood on (bold, opaque).
_BREEZE_MARK: Final[Color] = (112, 204, 255)
_STENCH_MARK: Final[Color] = (150, 226, 96)
_GLITTER_MARK: Final[Color] = (255, 226, 64)
# True-map layer (OBSERVER mode only): the same three hues, thin and translucent, on every cell.
_BREEZE_TRUE: Final[Paint] = (84, 160, 240, 160)
_STENCH_TRUE: Final[Paint] = (112, 204, 88, 190)
_GLITTER_TRUE: Final[Paint] = (250, 208, 52, 230)

_PERCEPT_CHIPS: Final[tuple[tuple[str, Color], ...]] = (
    ('breeze', (96, 196, 232)),
    ('stench', (146, 188, 72)),
    ('glitter', (252, 204, 44)),
    ('scream', (242, 110, 90)),
    ('bump', (186, 160, 220)),
)
_PERCEPT_COLORS: Final[dict[str, Color]] = dict(_PERCEPT_CHIPS)
_OUTCOME_COLORS: Final[dict[Outcome, Color]] = {
    Outcome.SUCCESS: _GOOD,
    Outcome.CLIMBED_WITHOUT_GOLD: _AMBER,
    Outcome.DIED: _BAD,
    Outcome.FORFEIT: _BAD,
    Outcome.ABORTED: _DIM,
}
_SCREEN_DIR: Final[dict[Direction, tuple[int, int]]] = {  # y grows downwards on the screen
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}
_CLOSE_KEYS: Final[frozenset[int]] = frozenset({pygame.K_ESCAPE, pygame.K_q})
_HEADLESS_DRIVERS: Final[frozenset[str]] = frozenset({'dummy', 'offscreen'})  # SDL drivers without a real window
# Window events after which the picture has to be put on the screen again (the names depend on the pygame version).
_REPAINT_EVENTS: Final[frozenset[int]] = frozenset(
    getattr(pygame, name)
    for name in ('VIDEOEXPOSE', 'WINDOWEXPOSED', 'WINDOWRESTORED', 'WINDOWSHOWN')
    if hasattr(pygame, name)
)
_REPAINT_S: Final[float] = 0.25  # pump() also repaints this often, in case such an event never arrives
_TRACE_ROWS: Final[int] = 12  # the trace log shows the latest 12 turns; there is no scrolling
_TRACE_START: Final[str] = 'START'  # action column of the frame before the first action


# ---- pure helpers ------------------------------------------------------------------------------------
def _clean(text: str) -> str:
    '''Post: `text` without control and invisible characters (titles and names may come from an agent).'''
    return ''.join(ch for ch in text if ch.isprintable())


def _shot_ray(frame: RenderFrame) -> list[Position]:
    '''Post: the cells crossed by the last SHOOT, nearest first, rebuilt from this frame alone: the ray from
    the agent in its facing direction to the boundary or, when the percept shows a scream, up to the nearest
    dead Wumpus on it. Empty unless the frame carries the true map and the last action was a SHOOT.
    '''
    world = frame.world
    if world is None or frame.last_action is not Action.SHOOT:
        return []
    scream = frame.last_percept is not None and frame.last_percept.scream
    facing = frame.agent.direction
    ray: list[Position] = []
    cell = frame.agent.position.neighbor(facing)
    while cell is not None:
        ray.append(cell)
        if scream and cell in world.wumpuses_dead:
            break
        cell = cell.neighbor(facing)
    return ray


def _status(frame: RenderFrame) -> tuple[str, Color]:
    '''Post: the status word (alive, dead, exited, finished) with the cause the frame carries, and a colour.'''
    agent = frame.agent
    if not agent.alive:
        word, color = 'dead', _BAD
    elif agent.exited:
        word, color = 'exited', _GOOD
    elif frame.terminal:
        word, color = 'finished', _AMBER
    else:
        word, color = 'alive', _GOOD
    if frame.termination is not None:
        return f'{word} ({frame.termination.value})', color
    if frame.outcome is not None:
        return f'{word} ({frame.outcome.value})', color
    return word, color


def _active_percepts(frame: RenderFrame) -> list[str]:
    '''Post: the names of the percepts that are on in the last observation.'''
    if frame.last_percept is None:
        return []
    return [name for name, on in frame.last_percept.to_wire().items() if on]


def _wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    '''Post: `text` broken into lines no wider than `width` pixels (a single over-long word is cut).'''
    lines: list[str] = []
    current = ''
    for word in text.split():
        while font.size(word)[0] > width and len(word) > 1:  # cut a word that cannot fit on any line
            cut = len(word) - 1
            while cut > 1 and font.size(word[:cut])[0] > width:
                cut -= 1
            if current:
                lines.append(current)
                current = ''
            lines.append(word[:cut])
            word = word[cut:]
        candidate = word if not current else f'{current} {word}'
        if font.size(candidate)[0] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _agent_points(center: Point, direction: Direction, radius: float) -> list[Point]:
    '''Post: the corners of the triangle that shows the agent; the tip points the way it faces.'''
    dx, dy = _SCREEN_DIR[direction]
    px, py = -dy, dx
    cx, cy = center
    bx, by = cx - dx * radius * 0.7, cy - dy * radius * 0.7
    return [
        (cx + dx * radius, cy + dy * radius),
        (bx + px * radius * 0.82, by + py * radius * 0.82),
        (bx - px * radius * 0.82, by - py * radius * 0.82),
    ]


# ---- sprites (shared by the board and the legend) -------------------------------------------------------
def _draw_pit(surface: pygame.Surface, center: Point, radius: float) -> None:
    pygame.draw.circle(surface, _PIT_RIM, center, radius)
    pygame.draw.circle(surface, _PIT_HOLE, center, max(radius - max(2.0, radius * 0.2), 1.0))


def _draw_wumpus(surface: pygame.Surface, center: Point, radius: float, *, alive: bool) -> None:
    cx, cy = center
    body = pygame.Rect(0, 0, int(2 * radius), int(2 * radius))
    body.center = (int(cx), int(cy))
    pygame.draw.rect(surface, _WUMPUS if alive else _WUMPUS_DEAD, body, border_radius=max(int(radius * 0.5), 2))
    eye_r = max(radius * 0.24, 2.0)
    for side in (-1, 1):
        eye = (cx + side * radius * 0.42, cy - radius * 0.2)
        if alive:
            pygame.draw.circle(surface, _WHITE, eye, eye_r)
            pygame.draw.circle(surface, _BLACK, eye, max(eye_r * 0.5, 1.0))
        else:  # dead eyes are crosses
            d = eye_r * 0.8
            pygame.draw.line(surface, _BLACK, (eye[0] - d, eye[1] - d), (eye[0] + d, eye[1] + d), 2)
            pygame.draw.line(surface, _BLACK, (eye[0] - d, eye[1] + d), (eye[0] + d, eye[1] - d), 2)
    mouth_y = cy + radius * 0.42
    if alive:  # jagged teeth
        xs = [cx - radius * 0.5 + radius * 0.25 * i for i in range(5)]
        zigzag = [(x, mouth_y + (radius * 0.16 if i % 2 else -radius * 0.04)) for i, x in enumerate(xs)]
        pygame.draw.lines(surface, _BLACK, False, zigzag, 2)
    else:
        pygame.draw.line(surface, _BLACK, (cx - radius * 0.45, mouth_y), (cx + radius * 0.45, mouth_y), 2)


def _draw_gold(surface: pygame.Surface, center: Point, radius: float, *, taken: bool) -> None:
    if taken:  # the cell where the gold lay: a faded ring
        pygame.draw.circle(surface, _GOLD_FADED, center, radius, width=2)
        return
    pygame.draw.circle(surface, _GOLD_RIM, center, radius)
    pygame.draw.circle(surface, _GOLD, center, max(radius - 3.0, 1.0))
    pygame.draw.circle(surface, _GOLD_SHINE, (center[0] - radius * 0.3, center[1] - radius * 0.3), max(radius * 0.2, 1.0))


def _draw_agent(surface: pygame.Surface, center: Point, direction: Direction, radius: float, *, color: Color) -> None:
    points = _agent_points(center, direction, radius)
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, _WHITE, points, width=2)


# ---- percept marks (shared by the two grid layers and the legend) -----------------------------------------
def _draw_breeze(surface: pygame.Surface, center: Point, radius: float, color: Paint, width: int) -> None:
    '''Post: breeze = a blue circle (outline) around `center`.'''
    pygame.draw.circle(surface, color, center, radius, width=width)


def _draw_stench(surface: pygame.Surface, center: Point, radius: float, color: Paint, width: int) -> None:
    '''Post: stench = a green diagonal cross through `center`; `radius` is half the side of its square.'''
    cx, cy = center
    pygame.draw.line(surface, color, (cx - radius, cy - radius), (cx + radius, cy + radius), width)
    pygame.draw.line(surface, color, (cx - radius, cy + radius), (cx + radius, cy - radius), width)


def _draw_glitter(surface: pygame.Surface, center: Point, radius: float, color: Paint, *, rim: bool) -> None:
    '''Post: glitter = a yellow dot at `center`; the agent layer adds a white rim.'''
    pygame.draw.circle(surface, color, center, radius)
    if rim:
        pygame.draw.circle(surface, _WHITE, center, radius, width=1)


def _true_percepts(cell: Position, world: WorldSnapshot) -> tuple[bool, bool, bool]:
    '''Post: (breeze, stench, glitter) of `cell` according to the true map, the rule World.perceive()
    applies: breeze if a 4-neighbour is a pit, stench if a 4-neighbour is a LIVING Wumpus, glitter if the
    gold lies here and was not taken.
    '''
    neighbours = cell.neighbors()
    breeze = any(other in world.pits for other in neighbours)
    stench = any(other in world.wumpuses_alive for other in neighbours)
    glitter = cell == world.gold_position and not world.gold_taken
    return breeze, stench, glitter


@dataclass(frozen=True, slots=True)
class _TraceEntry:
    '''One line of the trace log: Turn X | Action | Score | Percepts.'''

    turn: int  # actions taken so far; 0 for the frame before the first action
    action: str  # the action taken in this turn, or _TRACE_START
    score: int  # the total score after the action
    percepts: tuple[str, ...] | None  # what the agent perceived after the action; None: the game ended

    @classmethod
    def from_frame(cls, frame: RenderFrame) -> _TraceEntry:
        '''Post: the entry of `frame`; only plain immutable values are kept, never the frame itself.'''
        action = _TRACE_START if frame.last_action is None else frame.last_action.value
        percepts = None if frame.last_percept is None else tuple(_active_percepts(frame))
        return cls(frame.turn, action, frame.score_total, percepts)

    def percept_parts(self, *, short: bool = False) -> list[tuple[str, Color]]:
        '''Post: the percept column as coloured pieces, for example Breeze, Stench; 'none' when nothing was
        perceived and 'n/a (game over)' for the step that ended the game. With `short` every percept is cut
        to two letters (Br, St, Gl, Sc, Bu) for a line that would not fit otherwise.
        '''
        if self.percepts is None:
            return [('n/a' if short else 'n/a (game over)', _DIM)]
        if not self.percepts:
            return [('none', _DIM)]
        parts: list[tuple[str, Color]] = []
        for name in self.percepts:
            if parts:
                parts.append((', ', _DIM))
            parts.append(((name[:2] if short else name).capitalize(), _PERCEPT_COLORS.get(name, _TEXT)))
        return parts


@dataclass(frozen=True, slots=True)
class _Layout:
    '''Pixel geometry of the window, derived from the cell size.'''

    cell: int
    margin_left: int = 34
    margin_top: int = 12
    margin_bottom: int = 34
    gap: int = 14
    panel_width: int = 380  # wide enough for one trace log line: Turn X | Action | Score | Percepts
    pad: int = 12

    @property
    def grid_w(self) -> int:
        return self.cell * GRID_WIDTH

    @property
    def grid_h(self) -> int:
        return self.cell * GRID_HEIGHT

    @property
    def panel_x(self) -> int:
        return self.margin_left + self.grid_w + self.gap

    @property
    def window(self) -> tuple[int, int]:
        return (self.panel_x + self.panel_width + self.pad, self.margin_top + self.grid_h + self.margin_bottom)

    def rect(self, cell: Position) -> pygame.Rect:
        '''Post: the pixel rectangle of a cell; x grows to the right, y grows upwards, (1, 1) is bottom left.'''
        left = self.margin_left + (cell.x - MIN_COORD) * self.cell
        top = self.margin_top + (GRID_HEIGHT - cell.y) * self.cell
        return pygame.Rect(left, top, self.cell, self.cell)

    def center(self, cell: Position) -> Point:
        rect = self.rect(cell)
        return (rect.left + rect.width / 2, rect.top + rect.height / 2)


@dataclass(frozen=True, slots=True)
class _Fonts:
    title: pygame.font.Font
    big: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font
    trace: pygame.font.Font  # the rows of the trace log
    tiny: pygame.font.Font  # a trace log percept list too long for the trace font


class PygameRenderer(Renderer):
    '''Graphical renderer for the human Observer: a 20 x 20 board and a side panel in a pygame window.

    The side panel carries a trace log with the latest 12 turns (Turn X | Action | Score | Percepts).
    After the game the window stays open on the final board until the user closes it.

    The grid carries two layers of percept marks (breeze: blue circle, stench: green cross, glitter:
    yellow dot): what the agent perceived on the cells it stood on, in every mode, and - in OBSERVER mode
    only - the true percepts of every cell of the board.

    Keys: ESC or Q closes the window; closing it mid-game makes render() raise RendererError, which the
    GameEngine answers by switching to a NullRenderer so the game itself carries on unaffected.
    '''

    def __init__(
        self,
        *,
        cell_size: int = 36,
        step_delay_s: float = 0.15,
        end_hold_s: float | None = None,
        fps: int = 60,
        hold_on_end: bool = True,
    ) -> None:
        '''Pre: 16 <= cell_size <= 96; step_delay_s >= 0; end_hold_s is None or >= 0; 1 <= fps <= 240.
        Post: no window is opened yet. end_hold_s is accepted only so that existing calls keep working
        and is ignored: the final screen never closes on a timer, it stays until the user closes it.
        hold_on_end=True (the default) makes end() block until the user closes the window. A host with its
        own event loop passes False: end() then draws the summary and returns, the window stays open, and
        the host keeps it alive with pump() and closes it with close().
        Raises: ValueError: a parameter is out of range.
        '''
        if not 16 <= cell_size <= 96:
            raise ValueError('cell_size must be between 16 and 96 pixels')
        if not (math.isfinite(step_delay_s) and step_delay_s >= 0.0):
            raise ValueError('step_delay_s must be a finite number >= 0')
        if end_hold_s is not None and not (math.isfinite(end_hold_s) and end_hold_s >= 0.0):
            raise ValueError('end_hold_s must be None or a finite number >= 0')
        if not 1 <= fps <= 240:
            raise ValueError('fps must be between 1 and 240')
        self._layout = _Layout(cell=cell_size)
        self._step_delay_s = step_delay_s
        self._fps = fps
        self._hold_on_end = hold_on_end
        self._mode: RenderMode | None = None
        self._title = ''
        self._screen: pygame.Surface | None = None
        self._board: pygame.Surface | None = None
        self._fonts: _Fonts | None = None
        self._clock: pygame.time.Clock | None = None
        self._last_frame: RenderFrame | None = None
        self._trace: list[_TraceEntry] = []  # one entry per rendered frame of the current game
        self._percepts: dict[Position, Percept] = {}  # agent layer: latest percept on each cell it stood on
        self._true_world: WorldSnapshot | None = None  # the map the cached true-map layer was drawn for
        self._true_layer: pygame.Surface | None = None
        self._repaint = False  # a window event asked for the picture to be shown again; used by pump()
        self._flipped_at = 0.0  # time.monotonic() of pump()'s latest repaint

    # ---- context manager -----------------------------------------------------------------------------
    def __enter__(self) -> PygameRenderer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- window life cycle ---------------------------------------------------------------------------
    def _open_window(self) -> None:
        '''Raises: RendererError: the display or the fonts cannot be initialised (for example no display).'''
        try:
            pygame.display.init()
            pygame.font.init()
            self._screen = pygame.display.set_mode(self._layout.window)
            self._fonts = _Fonts(
                title=pygame.font.Font(None, 30),
                big=pygame.font.Font(None, 46),
                body=pygame.font.Font(None, 24),
                small=pygame.font.Font(None, 19),
                trace=pygame.font.Font(None, 18),
                tiny=pygame.font.Font(None, 15),
            )
            self._clock = pygame.time.Clock()
            self._board = self._render_board()
        except (pygame.error, OSError) as exc:
            self._teardown()
            raise RendererError(f'cannot open the pygame window: {exc}') from exc

    def _teardown(self) -> None:
        '''Post: the window is closed and every handle dropped; only display and font are shut down, so a
        host program that uses other pygame modules is left alone. Never raises.
        '''
        self._screen = self._board = self._fonts = self._clock = None
        self._mode = None
        self._last_frame = None
        self._trace = []
        self._percepts = {}
        self._true_world = self._true_layer = None
        try:
            pygame.display.quit()
            pygame.font.quit()
        except pygame.error:  # pragma: no cover - nothing sensible to do
            pass

    def _abort(self) -> NoReturn:
        '''The user closed the window: release everything and tell the engine the output is gone.'''
        self._teardown()
        raise RendererError('the pygame window was closed by the user')

    def _require_window(self) -> pygame.Surface:
        if self._screen is None or self._mode is None:
            raise RendererError('render() or end() was called before begin(), or after the window closed')
        return self._screen

    def _pump_events(self) -> bool:
        '''Post: True when the user asked to close the window (QUIT, ESC, Q). Draining the queue keeps the
        window responsive. An event that uncovered the window is remembered for pump().
        '''
        wanted = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                wanted = True
            elif event.type == pygame.KEYDOWN and event.key in _CLOSE_KEYS:
                wanted = True
            elif event.type in _REPAINT_EVENTS:
                self._repaint = True
        return wanted

    def _wait(self, seconds: float) -> None:
        '''Post: at least one event poll, then `seconds` of pacing with the window kept responsive.
        Raises: RendererError: the user closed the window meanwhile.
        '''
        clock = self._clock
        deadline = time.monotonic() + seconds
        while True:
            if self._pump_events():
                self._abort()
            if time.monotonic() >= deadline or clock is None:
                return
            clock.tick(self._fps)

    def _hold_open(self) -> None:
        '''Post: returns only once the user closes the window: its X button (pygame.QUIT) or ESC / Q. There
        is no timeout. The display is flipped on every tick, so the final board is repainted after the
        window was covered, moved or minimised. Without a real window (SDL's dummy or offscreen video
        driver, as used for headless runs) nobody could ever close it, so this returns at once.
        '''
        if pygame.display.get_driver() in _HEADLESS_DRIVERS:
            return
        clock = self._clock if self._clock is not None else pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in _CLOSE_KEYS:
                    return
            pygame.display.flip()
            clock.tick(self._fps)

    # ---- Renderer API --------------------------------------------------------------------------------
    def begin(self, title: str, mode: RenderMode) -> None:
        '''Pre: no game is currently displayed. Post: the window is open and shows the empty board; the
        trace log is empty.
        Raises: RendererError: the output is unavailable.
        '''
        if self._screen is None:
            self._open_window()
        self._mode = mode
        self._title = _clean(title)[:120]
        self._last_frame = None
        self._trace = []
        self._percepts = {}
        self._present(None)

    def render(self, frame: RenderFrame) -> None:
        '''Pre: begin() was called and frame.mode equals the mode given to begin().
        Post: the frame is drawn and added to the trace log, its percept is remembered for the cell the
        agent stands on (agent layer of the percept marks), the caption shows turn, score and percepts,
        then step_delay_s of responsive pacing follows.
        Raises: RendererError: begin() was not called, the pygame call failed or the window was closed.
        ValueError: mode mismatch.
        '''
        self._require_window()
        if frame.mode is not self._mode:
            mode = self._mode.value if self._mode is not None else '?'
            raise ValueError(f'frame mode {frame.mode.value} does not match the mode {mode} given to begin()')
        if self._pump_events():
            self._abort()
        self._trace.append(_TraceEntry.from_frame(frame))
        if frame.last_percept is not None:  # None: the step ended the game, nothing was perceived
            self._percepts[frame.agent.position] = frame.last_percept
        self._present(frame)
        self._wait(self._step_delay_s)

    def end(self, result: GameResult) -> None:
        '''Pre: begin() was called.
        Post: the final board (cells, path, agent) stays fully visible, with the summary and the trace log
        in the side panel, until the user closes the window (X button, ESC or Q); this call blocks until
        then and never closes the window on a timer. Afterwards the window is closed. In AGENT_VIEW mode
        only the coarse outcome, final score and actions taken are shown, never seeds, layouts or
        fine-grained causes.
        With hold_on_end=False nothing blocks: the summary is drawn and the call returns with the window
        still open; the host keeps calling pump() and decides when to close() or begin() the next game.
        Raises: RendererError: begin() was not called, or a pygame call failed.
        '''
        screen = self._require_window()
        teardown = self._hold_on_end
        try:
            self._draw(screen, self._last_frame, result)
            pygame.display.set_caption(self._caption(self._last_frame, result))
            pygame.display.flip()
            if self._hold_on_end:
                self._hold_open()
        except pygame.error as exc:
            teardown = True  # a window that cannot be drawn is not worth keeping
            raise RendererError(f'pygame failed while showing the final board: {exc}') from exc
        finally:
            if teardown:
                self._teardown()

    def close(self) -> None:
        '''Pre: none. Post: the window is closed; idempotent. Raises: nothing.'''
        self._teardown()

    @property
    def is_open(self) -> bool:
        '''Post: True while the window exists.'''
        return self._screen is not None

    def pump(self) -> bool:
        '''For a host program with its own event loop: call this a few dozen times per second.
        Pre: none. Post: never blocks. The pending window events are handled; the picture is put on the
        screen again when the window was uncovered, restored or shown, and a few times per second in any
        case, so the board survives being covered or moved without a full repaint on every call. Returns
        True while the window is open. When the user asked to close it (X button, ESC or Q) the window is
        closed and False is returned; False also when there is no window.
        Raises: nothing.
        '''
        if self._screen is None:
            return False
        try:
            if self._pump_events():
                self._teardown()
                return False
            now = time.monotonic()
            if self._repaint or now - self._flipped_at >= _REPAINT_S:
                pygame.display.flip()
                self._repaint = False
                self._flipped_at = now
        except pygame.error:
            self._teardown()
            return False
        return True

    def save_image(self, path: Path) -> None:
        '''Pre: a window is open. Post: its current contents are saved; the format follows the suffix.
        Raises: RendererError: no window is open, or the file cannot be written.
        '''
        screen = self._require_window()
        try:
            pygame.image.save(screen, str(path))
        except (pygame.error, OSError) as exc:
            raise RendererError(f'cannot save {path}: {exc}') from exc

    # ---- drawing -------------------------------------------------------------------------------------
    def _present(self, frame: RenderFrame | None) -> None:
        '''Draw `frame` (or the empty board), update the caption and flip the display.'''
        screen = self._require_window()
        try:
            self._draw(screen, frame)
            pygame.display.set_caption(self._caption(frame))
            pygame.display.flip()
        except pygame.error as exc:
            self._teardown()
            raise RendererError(f'pygame failed while drawing: {exc}') from exc
        self._last_frame = frame

    def _caption(self, frame: RenderFrame | None, result: GameResult | None = None) -> str:
        head = f'Wumpus World - {self._title}' if self._title else 'Wumpus World'
        if result is not None:
            return f'{head} | GAME OVER: {result.outcome.value} | score {result.final_score} | ESC closes'
        if frame is None:
            return head
        percepts = ', '.join(_active_percepts(frame)) or 'no percepts'
        return f'{head} | turn {frame.turn} | score {frame.score_total} | {percepts}'

    def _render_board(self) -> pygame.Surface:
        '''Post: the static background: checkered cells, grid lines and the axis labels.'''
        layout, fonts = self._layout, self._require_fonts()
        board = pygame.Surface(layout.window)
        board.fill(_BG)
        for y in range(MIN_COORD, GRID_HEIGHT + 1):
            for x in range(MIN_COORD, GRID_WIDTH + 1):
                color = _BOARD_A if (x + y) % 2 == 0 else _BOARD_B
                pygame.draw.rect(board, color, layout.rect(Position(x, y)))
        left, top = layout.margin_left, layout.margin_top
        for i in range(GRID_WIDTH + 1):
            pygame.draw.line(board, _GRID_LINE, (left + i * layout.cell, top), (left + i * layout.cell, top + layout.grid_h))
        for j in range(GRID_HEIGHT + 1):
            pygame.draw.line(board, _GRID_LINE, (left, top + j * layout.cell), (left + layout.grid_w, top + j * layout.cell))
        for n in range(MIN_COORD, GRID_WIDTH + 1):
            label = fonts.small.render(str(n), True, _LABEL)
            rect = layout.rect(Position(n, MIN_COORD))
            board.blit(label, label.get_rect(center=(rect.centerx, top + layout.grid_h + layout.margin_bottom // 2)))
        for n in range(MIN_COORD, GRID_HEIGHT + 1):
            label = fonts.small.render(str(n), True, _LABEL)
            rect = layout.rect(Position(MIN_COORD, n))
            board.blit(label, label.get_rect(center=(left // 2, rect.centery)))
        return board

    def _require_fonts(self) -> _Fonts:
        if self._fonts is None:
            raise RendererError('the pygame fonts are not initialised')
        return self._fonts

    def _draw(self, screen: pygame.Surface, frame: RenderFrame | None, result: GameResult | None = None) -> None:
        '''Post: the whole window repainted: board, visited cells, the true-map percept layer (OBSERVER
        mode), the agent's percept layer, sprites, trail, shot, agent and the side panel. With a `result`
        (the game is over) the panel shows the summary; the board is drawn exactly as before.
        '''
        if self._board is None:
            raise RendererError('the pygame board is not initialised')
        screen.blit(self._board, (0, 0))
        if frame is None:
            self._draw_panel(screen, None, result)
            return
        layout = self._layout
        agent, world = frame.agent, frame.world
        for cell in agent.visited:  # visited cells get a green tint
            color = _VISITED_A if (cell.x + cell.y) % 2 == 0 else _VISITED_B
            pygame.draw.rect(screen, color, layout.rect(cell))
        if world is not None:  # only a frame in OBSERVER mode carries the true map
            screen.blit(self._true_marks(world), (0, 0))
        self._draw_agent_marks(screen)
        radius = layout.cell * 0.36
        if world is not None:
            _draw_gold(screen, layout.center(world.gold_position), radius * 0.85, taken=world.gold_taken)
            for pit in world.pits:
                _draw_pit(screen, layout.center(pit), radius)
            for wumpus in world.wumpuses_alive:
                _draw_wumpus(screen, layout.center(wumpus), radius, alive=True)
            for wumpus in world.wumpuses_dead:
                _draw_wumpus(screen, layout.center(wumpus), radius, alive=False)
        if len(agent.path) >= 2:  # the path travelled
            pygame.draw.lines(screen, _TRAIL, False, [layout.center(cell) for cell in agent.path], 2)
        self._draw_shot(screen, frame)
        self._draw_agent_sprite(screen, frame)
        self._draw_panel(screen, frame, result)

    def _mark_geometry(self, cell: Position) -> tuple[Point, float, Point, float]:
        '''Post: where the percept marks of `cell` go, the same for both layers: the centre and radius of
        the breeze circle and the stench cross, then the centre and radius of the glitter dot in the top
        right corner of the cell (so it stays visible next to the gold sprite).
        '''
        size = self._layout.cell
        rect = self._layout.rect(cell)
        dot_r = max(size * 0.11, 2.5)
        dot = (rect.right - dot_r - 3, rect.top + dot_r + 3)
        return self._layout.center(cell), size * 0.4, dot, dot_r

    def _true_marks(self, world: WorldSnapshot) -> pygame.Surface:
        '''Post: the true-map layer of `world`, a transparent surface the size of the window: for every
        cell of the board a thin translucent blue circle if a 4-neighbour is a pit (breeze), a green cross
        if a 4-neighbour is a living Wumpus (stench) and a yellow dot if the untaken gold lies there
        (glitter). The layer is cached and redrawn only when the map changes (a Wumpus died, the gold was
        taken, a new game).
        '''
        if self._true_layer is not None and self._true_world == world:
            return self._true_layer
        layer = pygame.Surface(self._layout.window, pygame.SRCALPHA)
        width = 2 if self._layout.cell >= 28 else 1
        for y in range(MIN_COORD, GRID_HEIGHT + 1):
            for x in range(MIN_COORD, GRID_WIDTH + 1):
                cell = Position(x, y)
                breeze, stench, glitter = _true_percepts(cell, world)
                center, radius, dot, dot_r = self._mark_geometry(cell)
                if breeze:
                    _draw_breeze(layer, center, radius, _BREEZE_TRUE, width)
                if stench:
                    _draw_stench(layer, center, radius * 0.72, _STENCH_TRUE, width)
                if glitter:
                    _draw_glitter(layer, dot, dot_r, _GLITTER_TRUE, rim=False)
        self._true_world, self._true_layer = world, layer
        return layer

    def _draw_agent_marks(self, screen: pygame.Surface) -> None:
        '''Post: the agent layer: on every cell the agent stood on, the breeze, stench and glitter it last
        perceived there, bold and opaque. A mark is what the agent knows, so a stench stays as it was last
        perceived even if that Wumpus was shot later; the true-map layer shows the current truth.
        '''
        width = 3 if self._layout.cell >= 28 else 2
        for cell, percept in self._percepts.items():
            center, radius, dot, dot_r = self._mark_geometry(cell)
            if percept.breeze:
                _draw_breeze(screen, center, radius, _BREEZE_MARK, width)
            if percept.stench:
                _draw_stench(screen, center, radius * 0.72, _STENCH_MARK, width)
            if percept.glitter:
                _draw_glitter(screen, dot, dot_r + 1, _GLITTER_MARK, rim=True)

    def _draw_shot(self, screen: pygame.Surface, frame: RenderFrame) -> None:
        '''Post: an orange arrow along the last shot, with a ring on the Wumpus it killed.'''
        ray = _shot_ray(frame)
        world = frame.world
        if not ray or world is None:
            return
        layout = self._layout
        dx, dy = _SCREEN_DIR[frame.agent.direction]
        start = layout.center(frame.agent.position)
        tip = layout.center(ray[-1])
        reach = layout.cell * 0.32  # stop inside the last cell so the arrow head is visible
        end = (tip[0] + dx * reach, tip[1] + dy * reach)
        pygame.draw.line(screen, _SHOT, start, end, 4)
        px, py = -dy, dx
        size = layout.cell * 0.2
        head = [
            (end[0] + dx * size, end[1] + dy * size),
            (end[0] - dx * size * 0.4 + px * size, end[1] - dy * size * 0.4 + py * size),
            (end[0] - dx * size * 0.4 - px * size, end[1] - dy * size * 0.4 - py * size),
        ]
        pygame.draw.polygon(screen, _SHOT, head)
        if frame.last_percept is not None and frame.last_percept.scream and ray[-1] in world.wumpuses_dead:
            pygame.draw.circle(screen, _SHOT, tip, layout.cell * 0.46, width=3)

    def _draw_agent_sprite(self, screen: pygame.Surface, frame: RenderFrame) -> None:
        '''Post: the agent triangle (tip = facing). A dead agent is grey with a red cross, one that exited
        has a green frame, and a small gold disc marks that it carries the gold.
        '''
        layout, agent = self._layout, frame.agent
        center = layout.center(agent.position)
        radius = layout.cell * 0.38
        if not agent.alive:
            _draw_agent(screen, center, agent.direction, radius, color=_AGENT_DEAD)
            d = layout.cell * 0.3
            for a, b in (((-d, -d), (d, d)), ((-d, d), (d, -d))):
                pygame.draw.line(screen, _BAD, (center[0] + a[0], center[1] + a[1]), (center[0] + b[0], center[1] + b[1]), 4)
            return
        if agent.exited:
            pygame.draw.rect(screen, _GOOD, layout.rect(agent.position), width=3)
        _draw_agent(screen, center, agent.direction, radius, color=_AGENT)
        if agent.has_gold:
            pygame.draw.circle(screen, _GOLD, center, max(layout.cell * 0.1, 3.0))
            pygame.draw.circle(screen, _GOLD_RIM, center, max(layout.cell * 0.1, 3.0), width=1)

    def _text(self, screen: pygame.Surface, text: str, x: int, y: int, font: pygame.font.Font, color: Color) -> int:
        '''Post: `text` drawn at (x, y); returns the y just below it.'''
        screen.blit(font.render(text, True, color), (x, y))
        return y + font.get_linesize()

    def _draw_panel(self, screen: pygame.Surface, frame: RenderFrame | None, result: GameResult | None = None) -> None:
        '''Post: the side panel: title, mode, the current turn (or, with a `result`, the end-of-game
        summary), the trace log and the legend.
        '''
        layout, fonts = self._layout, self._require_fonts()
        panel = pygame.Rect(layout.panel_x, layout.margin_top, layout.panel_width, layout.grid_h)
        pygame.draw.rect(screen, _PANEL, panel, border_radius=10)
        x, width = panel.left + 14, panel.width - 28
        y = panel.top + 12
        for line in _wrap(self._title or 'Wumpus World', fonts.title, width)[:2]:
            y = self._text(screen, line, x, y, fonts.title, _TEXT)
        observer = self._mode is RenderMode.OBSERVER
        badge = 'OBSERVER VIEW (true map)' if observer else 'AGENT VIEW (fog of war)'
        y = self._text(screen, badge, x, y + 2, fonts.small, _AMBER if observer else _AGENT) + 8
        pygame.draw.line(screen, _GRID_LINE, (x, y), (x + width, y), 2)
        y += 10
        if result is not None:
            y = self._draw_summary(screen, result, frame, x, y, width, observer=observer)
        elif frame is None:
            y = self._text(screen, 'waiting for the first frame...', x, y, fonts.body, _DIM)
        else:
            y = self._draw_turn(screen, frame, x, y, width, observer=observer)
        legend_y = panel.bottom - 146
        self._draw_trace(screen, x, y, legend_y - 16, width)
        self._draw_legend(screen, x, legend_y, width, observer=observer, finished=result is not None)

    def _draw_turn(self, screen: pygame.Surface, frame: RenderFrame, x: int, y: int, width: int, *, observer: bool) -> int:
        '''Post: the state after the last action: turn, score, agent, status and the percept chips;
        returns the y below it.
        '''
        fonts = self._require_fonts()
        agent = frame.agent
        action = '-' if frame.last_action is None else frame.last_action.value
        y = self._text(screen, f'Turn {frame.turn}   {action}', x, y, fonts.body, _TEXT)
        score_color = _GOOD if frame.score_total > 0 else _TEXT
        screen.blit(fonts.big.render(f'Score {frame.score_total}', True, score_color), (x, y))
        delta_color = _GOOD if frame.score_delta > 0 else (_BAD if frame.score_delta < 0 else _DIM)
        delta = fonts.body.render(f'({frame.score_delta:+d})', True, delta_color)
        screen.blit(delta, (x + width - delta.get_width(), y + 14))
        y += fonts.big.get_linesize() + 4
        y = self._text(screen, f'Agent ({agent.position.x}, {agent.position.y}) facing {agent.direction.value}', x, y, fonts.body, _TEXT)
        status, status_color = _status(frame)
        y = self._text(screen, f'Arrows {agent.arrows}   Gold held: {"yes" if agent.has_gold else "no"}', x, y, fonts.body, _TEXT)
        y = self._text(screen, f'Status: {status}', x, y, fonts.body, status_color)
        y = self._text(screen, f'Path {max(len(agent.path) - 1, 0)} moves, {len(agent.visited)} cells visited', x, y, fonts.small, _DIM) + 6
        y = self._draw_chips(screen, frame, x, y, width) + 6
        world = frame.world
        if observer and world is not None:
            gold = world.gold_position
            state = 'taken' if world.gold_taken else ('found, not taken' if gold in agent.visited else 'not found')
            y = self._text(screen, f'Gold at ({gold.x}, {gold.y}): {state}', x, y, fonts.small, _GOLD) + 4
        if frame.note is not None:
            for line in _wrap(_clean(frame.note), fonts.small, width)[:3]:
                y = self._text(screen, line, x, y, fonts.small, _DIM)
        return y

    def _draw_chips(self, screen: pygame.Surface, frame: RenderFrame, x: int, y: int, width: int) -> int:
        '''Post: one chip per percept, lit when on in the last observation; returns the y below the chips.'''
        fonts = self._require_fonts()
        if frame.last_percept is None:
            return self._text(screen, 'Percepts: n/a (game over)', x, y, fonts.body, _DIM)
        active = set(_active_percepts(frame))
        gap, height = 6, 26
        columns = len(_PERCEPT_CHIPS)  # the panel is wide enough for one row, which leaves room for the trace log
        chip_w = (width - gap * (columns - 1)) // columns
        for index, (name, color) in enumerate(_PERCEPT_CHIPS):
            rect = pygame.Rect(x + index * (chip_w + gap), y, chip_w, height)
            on = name in active
            if on:
                pygame.draw.rect(screen, color, rect, border_radius=7)
            else:
                pygame.draw.rect(screen, _GRID_LINE, rect, width=2, border_radius=7)
            label = fonts.small.render(name.capitalize(), True, _BLACK if on else _DIM)
            screen.blit(label, label.get_rect(center=rect.center))
        return y + height

    def _draw_trace(self, screen: pygame.Surface, x: int, top: int, bottom: int, width: int) -> None:
        '''Post: the trace log, aligned to `bottom`: a header and one line per turn in the form
        Turn X | Action | Score | Percepts, oldest first, the newest highlighted. Only the latest
        _TRACE_ROWS turns are shown (a plain slice, no scrolling), fewer when a small window leaves less
        room below `top`.
        '''
        fonts = self._require_fonts()
        font = fonts.trace
        head_h = fonts.body.get_linesize() + fonts.small.get_linesize() + 4
        row_h = font.get_linesize() + 2
        rows = min(_TRACE_ROWS, (bottom - top - 10 - head_h) // row_h)
        if rows < 1:
            return  # a very small window has no room for the log
        shown = self._trace[-rows:]  # the latest turns only
        y = bottom - head_h - rows * row_h
        pygame.draw.line(screen, _GRID_LINE, (x, y - 8), (x + width, y - 8), 2)
        self._text(screen, 'TRACE LOG', x, y, fonts.body, _TEXT)
        counter = fonts.small.render(f'latest {len(shown)} of {len(self._trace)} entries', True, _DIM)
        screen.blit(counter, (x + width - counter.get_width(), y + 3))
        y += fonts.body.get_linesize()
        y = self._text(screen, 'Turn | Action | Score | Percepts after the action', x, y, fonts.small, _DIM) + 4
        if not shown:
            self._text(screen, 'no turns yet', x, y, font, _DIM)
            return
        separator = font.render(' | ', True, _DIM)
        turn_w = max(font.size(f'Turn {entry.turn}')[0] for entry in shown)
        action_w = max(font.size(name)[0] for name in (_TRACE_START, *Action.names()))
        score_w = max(font.size(str(entry.score))[0] for entry in shown)
        action_x = x + turn_w + separator.get_width()
        score_x = action_x + action_w + separator.get_width()
        percept_x = score_x + score_w + separator.get_width()
        for index, entry in enumerate(shown):
            newest = index == len(shown) - 1
            color = _TEXT if newest else _LABEL
            if newest:
                pygame.draw.rect(screen, _BOARD_B, pygame.Rect(x - 6, y, width + 12, row_h), border_radius=4)
            text_y = y + (row_h - font.get_height()) // 2
            screen.blit(font.render('Turn', True, color), (x, text_y))
            number = font.render(str(entry.turn), True, color)
            screen.blit(number, (x + turn_w - number.get_width(), text_y))
            screen.blit(separator, (x + turn_w, text_y))
            screen.blit(font.render(entry.action, True, color), (action_x, text_y))
            screen.blit(separator, (action_x + action_w, text_y))
            score = font.render(str(entry.score), True, color)
            screen.blit(score, (score_x + score_w - score.get_width(), text_y))
            screen.blit(separator, (score_x + score_w, text_y))
            self._draw_trace_percepts(screen, entry, pygame.Rect(percept_x, y, x + width - percept_x, row_h))
            y += row_h

    def _draw_trace_percepts(self, screen: pygame.Surface, entry: _TraceEntry, cell: pygame.Rect) -> None:
        '''Post: the percepts of `entry` inside `cell`, each in the colour of its chip. A list too wide for
        the trace font is drawn in the tiny font, and one too wide even for that with two-letter names, so
        no percept is ever dropped; the cell is clipped as a last resort.
        '''
        fonts = self._require_fonts()
        full, short = entry.percept_parts(), entry.percept_parts(short=True)
        for font, parts in ((fonts.trace, full), (fonts.tiny, full), (fonts.trace, short)):
            if font.size(''.join(text for text, _ in parts))[0] <= cell.width:
                break  # the first variant that fits; the last one is kept when none does
        previous_clip = screen.get_clip()
        screen.set_clip(cell.clip(previous_clip))
        x = cell.left
        for part, color in parts:
            surface = font.render(part, True, color)
            screen.blit(surface, (x, cell.top + (cell.height - surface.get_height()) // 2))
            x += surface.get_width()
        screen.set_clip(previous_clip)

    def _draw_legend(self, screen: pygame.Surface, x: int, y: int, width: int, *, observer: bool, finished: bool = False) -> None:
        '''Post: a three-column legend of the sprites and percept marks that can appear in this mode, a
        line on the two mark layers, and the key hint.
        '''
        fonts = self._require_fonts()
        entries: list[str] = ['agent', 'visited', 'breeze', 'stench', 'glitter']
        marks = 'Marks: what the agent perceived where it stood'
        if observer:
            entries = ['agent', 'pit', 'wumpus', 'gold', 'taken gold', 'dead wumpus',
                       'visited', 'shot path', 'breeze', 'stench', 'glitter']
            marks = 'Marks: thin = true map, bold = perceived by agent'
        pygame.draw.line(screen, _GRID_LINE, (x, y - 8), (x + width, y - 8), 2)
        column_w = width // 3
        for index, name in enumerate(entries):
            cx = x + (index % 3) * column_w + 10
            cy = y + (index // 3) * 24 + 10
            self._draw_legend_icon(screen, name, (cx, cy))
            self._text(screen, name, cx + 18, cy - 8, fonts.small, _TEXT)
        y = self._text(screen, marks, x, y + 4 * 24 + 6, fonts.small, _DIM)
        if finished:  # the window now waits for the user
            self._text(screen, 'Game over - press ESC / Q or close the window to exit', x, y, fonts.small, _AMBER)
        else:
            self._text(screen, 'ESC / Q closes the window', x, y, fonts.small, _DIM)

    @staticmethod
    def _draw_legend_icon(screen: pygame.Surface, name: str, center: Point) -> None:
        if name == 'agent':
            _draw_agent(screen, center, Direction.EAST, 8, color=_AGENT)
        elif name == 'pit':
            _draw_pit(screen, center, 8)
        elif name == 'wumpus':
            _draw_wumpus(screen, center, 8, alive=True)
        elif name == 'dead wumpus':
            _draw_wumpus(screen, center, 8, alive=False)
        elif name == 'gold':
            _draw_gold(screen, center, 8, taken=False)
        elif name == 'taken gold':
            _draw_gold(screen, center, 8, taken=True)
        elif name == 'visited':
            pygame.draw.rect(screen, _VISITED_B, pygame.Rect(int(center[0]) - 8, int(center[1]) - 8, 16, 16))
        elif name == 'shot path':
            pygame.draw.line(screen, _SHOT, (center[0] - 9, center[1]), (center[0] + 9, center[1]), 3)
        elif name == 'breeze':
            _draw_breeze(screen, center, 8, _BREEZE_MARK, 2)
        elif name == 'stench':
            _draw_stench(screen, center, 6, _STENCH_MARK, 2)
        elif name == 'glitter':
            _draw_glitter(screen, center, 4, _GLITTER_MARK, rim=True)

    def _draw_summary(self, screen: pygame.Surface, result: GameResult, frame: RenderFrame | None,
                      x: int, y: int, width: int, *, observer: bool) -> int:
        '''Post: the end-of-game summary in the side panel, so that the final board and the agent's path
        stay fully visible; returns the y below it. `frame` is the last frame shown, if any.
        '''
        fonts = self._require_fonts()
        outcome = result.outcome
        y = self._text(screen, 'GAME OVER', x, y, fonts.big, _TEXT)
        y = self._text(screen, outcome.value.replace('_', ' '), x, y, fonts.title, _OUTCOME_COLORS.get(outcome, _TEXT)) + 2
        if observer:  # the Observer may see the fine-grained cause and the details
            y = self._text(screen, result.termination.value, x, y, fonts.body, _DIM)
        y = self._text(screen, f'Final score  {result.final_score}', x, y + 4, fonts.title, _TEXT)
        y = self._text(screen, f'Actions taken  {result.actions_taken}', x, y, fonts.body, _TEXT)
        if observer:
            y = self._text(screen, f'Arrows used {result.arrows_used}   Wumpuses killed {result.wumpuses_killed}', x, y, fonts.body, _DIM)
            y = self._text(screen, f'Gold grabbed: {"yes" if result.gold_grabbed else "no"}', x, y, fonts.body, _DIM)
            if result.agent_name is not None:
                for line in _wrap(f'Agent: {_clean(result.agent_name)}', fonts.body, width)[:1]:
                    y = self._text(screen, line, x, y, fonts.body, _DIM)
            if result.fault_message is not None:
                for line in _wrap(f'Fault: {_clean(result.fault_message)}', fonts.small, width)[:3]:
                    y = self._text(screen, line, x, y, fonts.small, _BAD)
        if frame is not None:
            agent = frame.agent
            where = f'Ended at ({agent.position.x}, {agent.position.y}) after {max(len(agent.path) - 1, 0)} moves, {len(agent.visited)} cells visited'
            y = self._text(screen, where, x, y, fonts.small, _DIM)
        return y


# ---- demo --------------------------------------------------------------------------------------------
def _demo(argv: Sequence[str]) -> int:
    '''Play a short scripted game on a dummy world: python -m wumpus_engine.display.pygame_renderer'''
    from wumpus_engine.config.schema import SandboxLevel
    from wumpus_engine.core.agent_state import AgentSnapshot
    from wumpus_engine.core.results import RunInfo, ScoreBucket, TerminationReason
    from wumpus_engine.core.rules import ScoreEvent
    from wumpus_engine.core.world import WorldSnapshot
    from wumpus_sdk.percept import Percept

    parser = argparse.ArgumentParser(
        description='Pygame renderer demo with a dummy world (no engine needed). After the last frame the '
                    'window stays open until you close it (X button, ESC or Q).')
    parser.add_argument('--delay', type=float, default=0.25, help='seconds per frame (default 0.25)')
    parser.add_argument('--agent-view', action='store_true', help='fog of war: no true map')
    parser.add_argument('--shot', type=Path, default=None, help='save a PNG of the last frame (works headless)')
    args = parser.parse_args(list(argv))

    pits = frozenset({Position(3, 3), Position(7, 5), Position(10, 12)})
    alive = frozenset({Position(5, 7)})
    dead = frozenset({Position(9, 2)})
    gold = Position(9, 8)
    # The agent walks east along row 1, north up column 6, and shoots east along row 2 to kill the Wumpus at (9, 2).
    walk = [Position(x, 1) for x in range(1, 7)] + [Position(6, y) for y in range(2, 5)]
    mode = RenderMode.AGENT_VIEW if args.agent_view else RenderMode.OBSERVER

    def snapshot(i: int, facing: Direction, arrows: int = 1, gold_held: bool = False) -> AgentSnapshot:
        path = tuple(walk[: i + 1])
        return AgentSnapshot(path[-1], facing, arrows, gold_held, True, False, i, frozenset(path), path)

    def frame(i: int, facing: Direction, action: Action | None, percept: Percept | None, *,
              arrows: int = 1, taken: bool = False, shot_dead: bool = False, note: str | None = None) -> RenderFrame:
        world = None if args.agent_view else WorldSnapshot(
            pits, alive, dead if shot_dead else frozenset(), gold, taken)
        return RenderFrame(mode, i, snapshot(i, facing, arrows), action, percept, -i, -1, False, None, None, world, note)

    quiet = Percept(False, False, False, False, False)
    frames = [frame(0, Direction.EAST, None, quiet, note='dummy game: no engine involved')]
    for i in range(1, len(walk)):
        facing = Direction.EAST if walk[i].y == 1 else Direction.NORTH
        breeze = Percept(True, False, False, False, False) if walk[i] in (Position(3, 2), Position(6, 5)) else quiet
        frames.append(frame(i, facing, Action.MOVE_FORWARD, breeze))
    last = len(walk) - 1
    shooter = AgentSnapshot(Position(6, 2), Direction.EAST, 0, False, True, False, last + 1,
                            frozenset(walk[:6]) | {Position(6, 2)}, tuple(walk[:6]) + (Position(6, 2),))
    shot_world = None if args.agent_view else WorldSnapshot(pits, alive, dead, gold, False)
    frames.append(RenderFrame(mode, last + 1, shooter, Action.SHOOT, Percept(False, False, False, True, False),
                              -last - 12, -11, False, None, None, shot_world, 'SHOOT east along row 2'))
    result = GameResult(
        RunInfo('d' * 32, 'demo', 'dummy', 0, 0, SandboxLevel.PROCESS), TerminationReason.CLIMBED_WITH_GOLD, 976,
        {ScoreEvent.GOLD_GRAB: ScoreBucket(1, 1000)}, 24, 1, 1, True, 3.2, 'demo agent', None, None,
    )

    with PygameRenderer(step_delay_s=args.delay) as renderer:
        renderer.begin('Demo: dummy agent on a dummy world', mode)
        for item in frames:
            renderer.render(item)
        if args.shot is not None:
            renderer.save_image(args.shot)
        renderer.end(result)  # blocks until the window is closed
    return 0


if __name__ == '__main__':
    raise SystemExit(_demo(sys.argv[1:]))
