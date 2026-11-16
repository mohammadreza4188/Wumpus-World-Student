# wumpus_engine/gateway/sandbox.py
from __future__ import annotations

import re
import unicodedata
from typing import Final

_MAX_SCAN_CHARS: Final[int] = 1 << 20  # bound the work done on a hostile input
_DROPPED_CATEGORIES: Final[frozenset[str]] = frozenset({'Cc', 'Cf', 'Cs', 'Zl', 'Zp'})

# String-terminated escape sequences: OSC, DCS, SOS, PM, APC (ESC ] / P / X / ^ / _ ... BEL or ESC \).
_ESC_STRINGS: Final[re.Pattern[str]] = re.compile(r'\x1b[\]PX^_].*?(?:\x07|\x1b\\|\Z)', re.DOTALL)
# CSI sequences, 7-bit (ESC [) and 8-bit (U+009B) introducers.
_CSI: Final[re.Pattern[str]] = re.compile(r'(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]')
# Any remaining ESC sequence: optional intermediate bytes, then one final byte.
_ESC_OTHER: Final[re.Pattern[str]] = re.compile(r'\x1b[ -/]*[0-~]')


def sanitize_text(raw: bytes | str, *, max_chars: int) -> str:
    '''Pre: none.
    Post: ANSI escape sequences, control characters (including newlines) and invisible format characters
    removed; undecodable bytes replaced; length capped at max_chars (an empty string if max_chars <= 0).
    Applied to every string that comes from the agent.
    Raises: nothing.
    '''
    text = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
    text = text[:_MAX_SCAN_CHARS]
    text = _ESC_STRINGS.sub('', text)
    text = _CSI.sub('', text)
    text = _ESC_OTHER.sub('', text)
    cleaned = ''.join(ch for ch in text if unicodedata.category(ch) not in _DROPPED_CATEGORIES)
    return cleaned[: max(max_chars, 0)]