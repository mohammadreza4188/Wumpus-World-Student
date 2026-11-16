# wumpus_engine/display/renderer.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from wumpus_engine.core.agent_state import AgentSnapshot
from wumpus_engine.core.results import GameResult, TerminationReason
from wumpus_engine.core.world import WorldSnapshot
from wumpus_sdk.actions import Action
from wumpus_sdk.dto import Outcome
from wumpus_sdk.percept import Percept


class RenderMode(StrEnum):
    '''Which information a frame may carry.'''

    OBSERVER = 'OBSERVER'  # true map visible: instructor and Observer tools only
    AGENT_VIEW = 'AGENT_VIEW'  # fog of war: only what the agent knows; safe to show students


@dataclass(frozen=True, slots=True)
class RenderFrame:
    '''Immutable snapshot of one turn for a Renderer.'''

    mode: RenderMode
    turn: int  # actions taken so far
    agent: AgentSnapshot
    last_action: Action | None
    last_percept: Percept | None
    score_total: int
    score_delta: int
    terminal: bool
    outcome: Outcome | None  # coarse result; allowed in every mode
    termination: TerminationReason | None  # fine-grained cause; OBSERVER mode only
    world: WorldSnapshot | None  # true map; OBSERVER mode only
    note: str | None = None

    def __post_init__(self) -> None:
        '''Raises: ValueError: if mode is OBSERVER and world is None, or if mode is AGENT_VIEW and world or
        termination is not None.
        '''
        if self.mode is RenderMode.OBSERVER and self.world is None:
            raise ValueError('an OBSERVER frame needs the true map')
        if self.mode is RenderMode.AGENT_VIEW and (self.world is not None or self.termination is not None):
            raise ValueError('an AGENT_VIEW frame must not carry the true map or the fine-grained cause')


class Renderer(ABC):
    '''Shows a game to the human Observer. Has no way to reach the agent.'''

    @abstractmethod
    def begin(self, title: str, mode: RenderMode) -> None:
        '''Pre: no game is currently displayed. Post: ready to receive frames of `mode`.
        Raises: RendererError: the output is unavailable.
        '''

    @abstractmethod
    def render(self, frame: RenderFrame) -> None:
        '''Pre: begin() was called and frame.mode equals the mode given to begin().
        Post: the frame is shown or queued; it is not mutated and nothing beyond its own immutable
        snapshots is retained.
        Raises: RendererError: output failure. ValueError: mode mismatch.
        '''

    @abstractmethod
    def end(self, result: GameResult) -> None:
        '''Pre: begin() was called. Post: the final summary is shown; in AGENT_VIEW mode only the coarse
        outcome, final score and actions taken, never seeds, layouts or fine-grained causes.
        Raises: RendererError: output failure.
        '''

    def close(self) -> None:
        '''Pre: none. Post: terminal or GUI handles released; idempotent; the default does nothing.
        Raises: nothing.
        '''
        return None


class NullRenderer(Renderer):
    '''Discards everything. The default for batch and evaluation runs.'''

    def begin(self, title: str, mode: RenderMode) -> None:
        '''Pre: none. Post: nothing happens. Raises: nothing.'''
        return None

    def render(self, frame: RenderFrame) -> None:
        '''Pre: none. Post: nothing happens. Raises: nothing.'''
        return None

    def end(self, result: GameResult) -> None:
        '''Pre: none. Post: nothing happens. Raises: nothing.'''
        return None