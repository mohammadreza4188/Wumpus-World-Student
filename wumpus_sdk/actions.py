# wumpus_sdk/actions.py
from __future__ import annotations

from enum import StrEnum

from wumpus_sdk.errors import InvalidActionError


class Action(StrEnum):
    '''The six legal actions. Member values equal member names.'''

    TURN_LEFT = 'TURN_LEFT'
    TURN_RIGHT = 'TURN_RIGHT'
    MOVE_FORWARD = 'MOVE_FORWARD'
    SHOOT = 'SHOOT'
    GRAB = 'GRAB'
    CLIMB = 'CLIMB'

    @classmethod
    def parse(cls, raw: object) -> Action:
        '''Pre: none. Post: the member whose name equals `raw` exactly (str, upper case, no whitespace).
        Raises: InvalidActionError: for anything else.
        '''
        if not isinstance(raw, str):
            raise InvalidActionError(f'an action must be a str, got {type(raw).__name__}')
        try:
            return cls(raw)
        except ValueError:
            raise InvalidActionError(f'unknown action {raw!r}') from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        '''Pre: none. Post: the six names in declaration order. Raises: nothing.'''
        return tuple(member.value for member in cls)