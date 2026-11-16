# wumpus_engine/core/errors.py
from __future__ import annotations

from wumpus_sdk.dto import AgentErrorDTO


class WumpusEngineError(Exception):
    '''Base class of every engine exception.'''


class ConfigError(WumpusEngineError):
    '''Configuration problem.'''


class ConfigNotFoundError(ConfigError):
    '''A configuration file or directory is missing.'''


class ConfigValidationError(ConfigError):
    '''A configuration rule is violated; `issues` lists every problem found.'''

    def __init__(self, message: str, issues: tuple[str, ...]) -> None:
        super().__init__(message)
        self.issues = issues


class ConfigSecurityError(ConfigError):
    '''A configuration path or id tries to escape its allowed root.'''


class WorldError(WumpusEngineError):
    '''World problem.'''


class InvalidWorldError(WorldError):
    '''A world layout violates an invariant.'''


class WorldGenerationError(WorldError):
    '''No valid world could be generated.'''


class EngineStateError(WumpusEngineError):
    '''An engine method was called in the wrong state.'''


class ScoreError(WumpusEngineError):
    '''Score problem.'''


class ScoreFinalizedError(ScoreError):
    '''The score ledger is finalized.'''


class LogError(WumpusEngineError):
    '''Action log problem.'''


class LogStateError(LogError):
    '''The log is closed, full, or the turn number is wrong.'''


class LogFormatError(LogError):
    '''A saved log is malformed.'''


class LogViewError(LogError):
    '''A log view cannot be produced.'''


class RendererError(WumpusEngineError):
    '''A renderer failed.'''


class GatewayError(WumpusEngineError):
    '''Agent gateway problem.'''


class GatewayStateError(GatewayError):
    '''A gateway method was called in the wrong state.'''


class ForbiddenPayloadError(GatewayError):
    '''An object that is not an allowed DTO was offered to the channel.'''


class AgentFault(GatewayError):
    '''The agent or its sandbox failed in a way that is charged to the agent.'''


class AgentLaunchError(AgentFault):
    '''The agent process could not be started or did not complete the handshake.'''

    def __init__(self, message: str, *, infrastructure: bool) -> None:
        super().__init__(message)
        self.infrastructure = infrastructure  # True: launcher or sandbox failed, not the student's fault


class AgentTimeoutError(AgentFault):
    '''No reply before the deadline.'''


class AgentCrashedError(AgentFault):
    '''The agent process ended or its pipe broke.'''

    def __init__(self, message: str, *, exit_code: int | None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class AgentReportedError(AgentFault):
    '''The agent reported an exception from student code.'''

    def __init__(self, dto: AgentErrorDTO) -> None:
        super().__init__(dto.message)
        self.dto = dto


class ProtocolViolationError(AgentFault):
    '''The agent sent a bad, oversized, stray or wrongly typed frame.'''


class EvaluatorError(WumpusEngineError):
    '''Evaluator problem.'''


class SuiteValidationError(EvaluatorError):
    '''An evaluation suite is invalid.'''


class SandboxUnavailableError(EvaluatorError):
    '''The configured sandbox level is not available on this machine.'''


class EvaluationAbortedError(EvaluatorError):
    '''The whole batch is invalid because sandbox infrastructure failed.'''