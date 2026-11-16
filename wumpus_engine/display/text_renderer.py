# wumpus_engine/display/text_renderer.py
from __future__ import annotations

import math
import sys
import time
from typing import Final, TextIO

from wumpus_engine.core.errors import RendererError
from wumpus_engine.core.results import GameResult
from wumpus_engine.display.renderer import RenderFrame, RenderMode, Renderer
from wumpus_sdk.actions import Action
from wumpus_sdk.types import GRID_HEIGHT, GRID_WIDTH, MIN_COORD, Direction, Position

_AGENT_GLYPHS: Final[dict[Direction, str]] = {
    Direction.NORTH: '^',
    Direction.EAST: '>',
    Direction.SOUTH: 'v',
    Direction.WEST: '<',
}
_LEGEND_OBSERVER: Final[str] = (
    'Legend: ^ > v < agent, P pit, W living Wumpus, x dead Wumpus, G gold, g taken gold, '
    '. visited cell, + last shot path'
)
_LEGEND_AGENT_VIEW: Final[str] = 'Legend: ^ > v < agent, . visited cell'
_SGR: Final[dict[str, str]] = {  # ANSI colours, used only when color=True
    '^': '1;32', '>': '1;32', 'v': '1;32', '<': '1;32',
    'P': '31', 'W': '1;35', 'x': '35', 'G': '1;33', 'g': '33', '+': '36', '.': '2',
}
_TRAIL_CELLS: Final[int] = 6


def _clean(text: str) -> str:
    '''Post: `text` without control and invisible characters (titles and names may come from an agent).'''
    return ''.join(ch for ch in text if ch.isprintable())


def _shot_cells(frame: RenderFrame) -> frozenset[Position]:
    '''Post: the cells crossed by the last SHOOT, rebuilt from this frame alone: the ray from the agent in its
    facing direction to the boundary, or, when the percept shows a scream, up to the nearest dead Wumpus
    on the ray. A frame cannot tell a SHOOT without arrows from a miss, so that case is drawn as a miss.
    '''
    world = frame.world
    if world is None or frame.last_action is not Action.SHOOT:
        return frozenset()
    scream = frame.last_percept is not None and frame.last_percept.scream
    facing = frame.agent.direction
    cells: list[Position] = []
    cell = frame.agent.position.neighbor(facing)
    while cell is not None:
        cells.append(cell)
        if scream and cell in world.wumpuses_dead:
            break
        cell = cell.neighbor(facing)
    return frozenset(cells)


def _status(frame: RenderFrame) -> str:
    '''Post: alive, dead, exited or finished, with the cause when the frame carries one.'''
    agent = frame.agent
    if not agent.alive:
        state = 'dead'
    elif agent.exited:
        state = 'exited'
    elif frame.terminal:
        state = 'finished'
    else:
        state = 'alive'
    if frame.termination is not None:
        return f'{state} ({frame.termination.value})'
    if frame.outcome is not None:
        return f'{state} ({frame.outcome.value})'
    return state


def _percepts(frame: RenderFrame) -> str:
    '''Post: the active percepts of the last action, 'none', or 'n/a' when the game ended.'''
    if frame.last_percept is None:
        return 'n/a'
    active = [name for name, on in frame.last_percept.to_wire().items() if on]
    return ', '.join(active) if active else 'none'


def _format_result(result: GameResult, mode: RenderMode) -> str:
    '''Post: the end-of-game summary. AGENT_VIEW shows only the coarse outcome, final score and actions.'''
    if mode is RenderMode.AGENT_VIEW:
        return '\n'.join(
            [
                '=== Game over ===',
                f'Outcome: {result.outcome.value}',
                f'Final score: {result.final_score}',
                f'Actions taken: {result.actions_taken}',
            ]
        )
    lines = [
        '=== Game over ===',
        f'Outcome: {result.outcome.value} ({result.termination.value})',
        f'Final score: {result.final_score}',
        f'Actions taken: {result.actions_taken}',
        f'Arrows used: {result.arrows_used}  Wumpuses killed: {result.wumpuses_killed}  '
        f'Gold grabbed: {"yes" if result.gold_grabbed else "no"}',
    ]
    if result.agent_name is not None:
        lines.append(f'Agent: {_clean(result.agent_name)}')
    if result.fault_message is not None:
        lines.append(f'Fault: {_clean(result.fault_message)}')
    return '\n'.join(lines)


class TextRenderer(Renderer):
    '''ASCII renderer for the human Observer. Writes to a text stream; never reaches the agent.'''

    def __init__(
        self, stream: TextIO | None = None, *, color: bool = False, step_delay_s: float = 0.0
    ) -> None:
        '''Pre: stream is writable (default: sys.stdout of the ENGINE process); step_delay_s >= 0.
        Post: nothing is written yet. Raises: ValueError: step_delay_s is negative or not finite.
        '''
        if not (math.isfinite(step_delay_s) and step_delay_s >= 0.0):
            raise ValueError('step_delay_s must be a finite number >= 0')
        self._stream = stream
        self._color = color
        self._step_delay_s = step_delay_s
        self._mode: RenderMode | None = None

    def _paint(self, glyph: str) -> str:
        code = _SGR.get(glyph)
        if not self._color or code is None:
            return glyph
        return f'\x1b[{code}m{glyph}\x1b[0m'

    def _glyph(self, frame: RenderFrame, cell: Position, shot: frozenset[Position]) -> str:
        '''Post: one map character. Priority: agent, pit, living Wumpus, dead Wumpus, gold, shot path,
        visited cell, blank. A dead agent shows what killed it instead of a facing arrow.
        '''
        agent = frame.agent
        world = frame.world
        if cell == agent.position and (agent.alive or world is None):
            return self._paint(_AGENT_GLYPHS[agent.direction])
        if world is not None:
            if cell in world.pits:
                return self._paint('P')
            if cell in world.wumpuses_alive:
                return self._paint('W')
            if cell in world.wumpuses_dead:
                return self._paint('x')
            if cell == world.gold_position:
                return self._paint('g' if world.gold_taken else 'G')
            if cell in shot:
                return self._paint('+')
        if cell in agent.visited:
            return self._paint('.')
        return ' '

    def format_frame(self, frame: RenderFrame) -> str:
        '''Pre: none.
        Post: ASCII status lines and the 20 x 20 map for one frame, north at the top and (1, 1) at the bottom
        left, without a trailing newline. Pure and deterministic, so golden-file tests can pin it; no I/O.
        In AGENT_VIEW mode only the agent, visited cells and the last percepts are drawn.
        Raises: nothing.
        '''
        agent = frame.agent
        world = frame.world
        observer = frame.mode is RenderMode.OBSERVER and world is not None
        action = '-' if frame.last_action is None else frame.last_action.value
        held = 'yes' if agent.has_gold else 'no'
        trail = ' '.join(f'({cell.x},{cell.y})' for cell in agent.path[-_TRAIL_CELLS:])
        if len(agent.path) > _TRAIL_CELLS:
            trail = f'... {trail}'
        lines = [
            f'Turn {frame.turn}  Action: {action}  Score: {frame.score_total} ({frame.score_delta:+d})',
            f'Agent: ({agent.position.x}, {agent.position.y}) facing {agent.direction.value}  '
            f'Arrows: {agent.arrows}  Gold held: {held}  Status: {_status(frame)}',
            f'Percepts: {_percepts(frame)}',
            f'Path: {max(len(agent.path) - 1, 0)} moves, {len(agent.visited)} cells visited, trail: {trail}',
        ]
        if observer and world is not None:
            gold = world.gold_position
            if world.gold_taken:
                state = 'taken'
            elif gold in agent.visited:
                state = 'found, not taken'
            else:
                state = 'not found'
            lines.append(f'Gold: at ({gold.x}, {gold.y}), {state}')
        if frame.note is not None:
            lines.append(f'Note: {_clean(frame.note)}')
        lines.append('')
        shot = _shot_cells(frame) if observer else frozenset()
        columns = range(MIN_COORD, GRID_WIDTH + 1)
        for y in range(GRID_HEIGHT, MIN_COORD - 1, -1):
            cells = ' '.join(self._glyph(frame, Position(x, y), shot) for x in columns)
            lines.append(f'{y:>2} {cells}'.rstrip())
        lines.append(('   ' + ' '.join(str(x // 10) if x >= 10 else ' ' for x in columns)).rstrip())
        lines.append('   ' + ' '.join(str(x % 10) for x in columns))
        lines.append(_LEGEND_OBSERVER if observer else _LEGEND_AGENT_VIEW)
        return '\n'.join(lines)

    def _write(self, text: str) -> None:
        '''Raises: RendererError: the stream cannot be written or flushed.'''
        stream = sys.stdout if self._stream is None else self._stream
        try:
            stream.write(text)
            stream.flush()
        except (OSError, ValueError) as exc:
            raise RendererError(f'cannot write to the output stream: {exc}') from exc

    def begin(self, title: str, mode: RenderMode) -> None:
        '''Pre: no game is currently displayed. Post: the banner is written and frames of `mode` are accepted.
        Raises: RendererError: the output is unavailable.
        '''
        self._write(f'=== {_clean(title)} [{mode.value}] ===\n\n')
        self._mode = mode

    def render(self, frame: RenderFrame) -> None:
        '''Pre: begin() was called and frame.mode equals the mode given to begin().
        Post: the formatted frame and a blank line are written, then the renderer sleeps step_delay_s.
        Raises: RendererError: begin() was not called, or the output failed. ValueError: mode mismatch.
        '''
        if self._mode is None:
            raise RendererError('render() was called before begin()')
        if frame.mode is not self._mode:
            raise ValueError(
                f'frame mode {frame.mode.value} does not match the mode {self._mode.value} given to begin()'
            )
        self._write(self.format_frame(frame) + '\n\n')
        if self._step_delay_s > 0.0:
            time.sleep(self._step_delay_s)

    def end(self, result: GameResult) -> None:
        '''Pre: begin() was called.
        Post: the summary is written; in AGENT_VIEW mode only the coarse outcome, final score and actions
        taken, never seeds, layouts or fine-grained causes. The game is no longer displayed.
        Raises: RendererError: begin() was not called, or the output failed.
        '''
        if self._mode is None:
            raise RendererError('end() was called before begin()')
        mode = self._mode
        self._mode = None
        self._write(_format_result(result, mode) + '\n')

    def close(self) -> None:
        '''Pre: none. Post: the stream is flushed, never closed; idempotent. Raises: nothing.'''
        stream = sys.stdout if self._stream is None else self._stream
        try:
            stream.flush()
        except (OSError, ValueError):
            pass