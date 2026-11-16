# wumpus_sdk/percept.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from wumpus_sdk.errors import ProtocolError

_FIELDS: Final[tuple[str, ...]] = ('breeze', 'stench', 'glitter', 'scream', 'bump')


@dataclass(frozen=True, slots=True)
class Percept:
    '''The five sensory flags of one observation.'''

    breeze: bool
    stench: bool
    glitter: bool
    scream: bool
    bump: bool

    def __post_init__(self) -> None:
        '''Raises: TypeError: if any field is not exactly a bool.'''
        for name in _FIELDS:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f'Percept.{name} must be a bool')

    def to_wire(self) -> dict[str, bool]:
        '''Pre: none. Post: JSON-safe object with the five boolean keys. Raises: nothing.'''
        return {
            'breeze': self.breeze,
            'stench': self.stench,
            'glitter': self.glitter,
            'scream': self.scream,
            'bump': self.bump,
        }

    @classmethod
    def from_wire(cls, raw: object) -> Percept:
        '''Raises: ProtocolError: unless raw is a dict with exactly the five keys, all JSON booleans.'''
        if not isinstance(raw, dict) or set(raw) != set(_FIELDS):
            raise ProtocolError('a percept must be an object with exactly the five percept keys')
        for name in _FIELDS:
            if type(raw[name]) is not bool:
                raise ProtocolError(f'percept.{name} must be a JSON boolean')
        return cls(
            breeze=raw['breeze'],
            stench=raw['stench'],
            glitter=raw['glitter'],
            scream=raw['scream'],
            bump=raw['bump'],
        )