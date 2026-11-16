# wumpus_sdk/errors.py
from __future__ import annotations


class WumpusSdkError(Exception):
    '''Base class of every exception raised by wumpus_sdk.'''


class ProtocolError(WumpusSdkError):
    '''A frame or DTO is malformed, oversized, of an unexpected type, or has unknown or missing fields.'''


class InvalidActionError(WumpusSdkError, ValueError):
    '''A value is not one of the six Action names.'''