# wumpus_engine/gateway/agent_gateway.py
from __future__ import annotations

from enum import StrEnum
from typing import Protocol
from wumpus_engine.config.schema import SandboxLevel
from wumpus_engine.core.errors import (
    AgentCrashedError,
    AgentLaunchError,
    AgentReportedError,
    AgentTimeoutError,
    ForbiddenPayloadError,
    GatewayStateError,
    ProtocolViolationError,
)
from wumpus_engine.core.results import GatewayReport
from wumpus_engine.gateway.sandbox import sanitize_text
from wumpus_engine.gateway.subprocess_launcher import SubprocessLauncher
from wumpus_sdk.codec import dto_from_json, dto_to_json
from wumpus_sdk.dto import (
    ActionDTO,
    AgentErrorDTO,
    GameOverDTO,
    HelloDTO,
    InvalidActionDTO,
    ObservationDTO,
    SessionInitDTO,
)
from wumpus_sdk.errors import ProtocolError


class GatewayState(StrEnum):
    """Lifecycle of one agent connection."""
    INIT = "INIT"
    OPEN = "OPEN"
    SESSION = "SESSION"
    FINISHED = "FINISHED"
    CLOSED = "CLOSED"


class AgentPort(Protocol):
    """What GameEngine needs from an agent connection."""
    @property
    def isolation_level(self) -> SandboxLevel:
        ...

    def open(self) -> HelloDTO:
        ...

    def begin_session(self, init: SessionInitDTO) -> None:
        ...

    def request_action(
        self, observation: ObservationDTO, *, timeout_s: float
    ) -> ActionDTO:
        ...

    def request_replacement_action(
        self, notice: InvalidActionDTO, *, timeout_s: float
    ) -> ActionDTO:
        ...

    def end_session(self, result: GameOverDTO) -> None:
        ...

    def close(self) -> GatewayReport:
        ...


class AgentGateway(AgentPort):
    """Secure subprocess gateway enforcing isolation and JSON DTO RPC."""

    def __init__(self, launcher: SubprocessLauncher) -> None:
        self._launcher = launcher
        self._state = GatewayState.INIT
        self._killed_by_engine = False
        self._frames_sent = 0
        self._frames_received = 0
        self._bytes_sent = 0
        self._bytes_received = 0

    @property
    def isolation_level(self) -> SandboxLevel:
        return self._launcher.isolation_level

    def open(self) -> HelloDTO:
        if self._state is not GatewayState.INIT:
            raise GatewayStateError(f"Cannot open in state {self._state}")

        try:
            self._launcher.launch()
        except Exception as exc:
            raise AgentLaunchError(f"Failed to launch subprocess: {exc}", infrastructure=True) from exc

        try:
            raw_line, nbytes = self._launcher.read_line(timeout_s=10.0)
        except TimeoutError:
            self._killed_by_engine = True
            self._launcher.terminate()
            raise AgentTimeoutError("Timeout waiting for HelloDTO during launch")

        if raw_line is None:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentLaunchError(
                f"Agent process exited before sending HelloDTO. Stderr: {stderr}",
                infrastructure=False,
            )

        self._frames_received += 1
        self._bytes_received += nbytes

        try:
            dto = dto_from_json(raw_line)
        except ProtocolError as exc:
            raise ProtocolViolationError(f"Invalid HelloDTO protocol: {exc}") from exc

        if isinstance(dto, AgentErrorDTO):
            raise AgentReportedError(dto)

        if not isinstance(dto, HelloDTO):
            raise ProtocolViolationError(f"Expected HelloDTO, got {type(dto)}")

        self._state = GatewayState.OPEN
        return dto

    def begin_session(self, init: SessionInitDTO) -> None:
        if self._state is not GatewayState.OPEN:
            raise GatewayStateError(f"Cannot begin session in state {self._state}")

        if not isinstance(init, SessionInitDTO):
            raise ForbiddenPayloadError(f"Expected SessionInitDTO, got {type(init)}")

        raw_json = dto_to_json(init)
        try:
            nbytes = self._launcher.write_line(raw_json)
            self._frames_sent += 1
            self._bytes_sent += nbytes
        except Exception as exc:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentCrashedError(
                f"Failed to write SessionInitDTO: {exc}. Stderr: {stderr}",
                exit_code=self._launcher.proc.poll() if self._launcher.proc else None,
            ) from exc

        self._state = GatewayState.SESSION

    def request_action(
        self, observation: ObservationDTO, *, timeout_s: float
    ) -> ActionDTO:
        if self._state is not GatewayState.SESSION:
            raise GatewayStateError(f"Cannot request action in state {self._state}")

        if not isinstance(observation, ObservationDTO):
            raise ForbiddenPayloadError(f"Expected ObservationDTO, got {type(observation)}")

        raw_json = dto_to_json(observation)
        try:
            nbytes = self._launcher.write_line(raw_json)
            self._frames_sent += 1
            self._bytes_sent += nbytes
        except Exception as exc:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentCrashedError(
                f"Child stdin closed: {exc}. Stderr: {stderr}",
                exit_code=self._launcher.proc.poll() if self._launcher.proc else None,
            ) from exc

        try:
            raw_line, nbytes = self._launcher.read_line(timeout_s=timeout_s)
        except TimeoutError:
            self._killed_by_engine = True
            self._launcher.terminate()
            raise AgentTimeoutError(f"Agent action timed out after {timeout_s:.2f}s")

        if raw_line is None:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentCrashedError(
                f"Agent process terminated unexpectedly. Stderr: {stderr}",
                exit_code=self._launcher.proc.poll() if self._launcher.proc else None,
            )

        self._frames_received += 1
        self._bytes_received += nbytes

        try:
            dto = dto_from_json(raw_line)
        except ProtocolError as exc:
            raise ProtocolViolationError(f"Invalid JSON frame from agent: {exc}") from exc

        if isinstance(dto, AgentErrorDTO):
            raise AgentReportedError(dto)

        if not isinstance(dto, ActionDTO):
            raise ProtocolViolationError(f"Expected ActionDTO, got {type(dto)}")

        return dto

    def request_replacement_action(
        self, notice: InvalidActionDTO, *, timeout_s: float
    ) -> ActionDTO:
        if self._state is not GatewayState.SESSION:
            raise GatewayStateError(f"Cannot request replacement action in state {self._state}")

        if not isinstance(notice, InvalidActionDTO):
            raise ForbiddenPayloadError(f"Expected InvalidActionDTO, got {type(notice)}")

        raw_json = dto_to_json(notice)
        try:
            nbytes = self._launcher.write_line(raw_json)
            self._frames_sent += 1
            self._bytes_sent += nbytes
        except Exception as exc:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentCrashedError(
                f"Child stdin closed during replacement: {exc}. Stderr: {stderr}",
                exit_code=self._launcher.proc.poll() if self._launcher.proc else None,
            ) from exc

        try:
            raw_line, nbytes = self._launcher.read_line(timeout_s=timeout_s)
        except TimeoutError:
            self._killed_by_engine = True
            self._launcher.terminate()
            raise AgentTimeoutError(f"Replacement action timed out after {timeout_s:.2f}s")

        if raw_line is None:
            stderr = sanitize_text(self._launcher.get_stderr_tail(), max_chars=1000)
            raise AgentCrashedError(
                f"Agent process terminated during replacement. Stderr: {stderr}",
                exit_code=self._launcher.proc.poll() if self._launcher.proc else None,
            )

        self._frames_received += 1
        self._bytes_received += nbytes

        try:
            dto = dto_from_json(raw_line)
        except ProtocolError as exc:
            raise ProtocolViolationError(f"Invalid JSON frame from agent: {exc}") from exc

        if isinstance(dto, AgentErrorDTO):
            raise AgentReportedError(dto)

        if not isinstance(dto, ActionDTO):
            raise ProtocolViolationError(f"Expected ActionDTO, got {type(dto)}")

        return dto

    def end_session(self, result: GameOverDTO) -> None:
        if self._state not in (GatewayState.OPEN, GatewayState.SESSION):
            raise GatewayStateError(f"Cannot end session in state {self._state}")

        if not isinstance(result, GameOverDTO):
            raise ForbiddenPayloadError(f"Expected GameOverDTO, got {type(result)}")

        try:
            raw_json = dto_to_json(result)
            self._launcher.write_line(raw_json)
            self._frames_sent += 1
        except Exception:
            pass
        finally:
            self._state = GatewayState.FINISHED

    def close(self) -> GatewayReport:
        if self._state is not GatewayState.CLOSED:
            self._launcher.terminate()
            self._state = GatewayState.CLOSED

        exit_code = self._launcher.proc.poll() if self._launcher.proc is not None else None
        stderr_tail = sanitize_text(self._launcher.get_stderr_tail(), max_chars=4096)

        return GatewayReport(
            exit_code=exit_code,
            killed_by_engine=self._killed_by_engine,
            stderr_tail=stderr_tail,
            frames_sent=self._frames_sent,
            frames_received=self._frames_received,
            bytes_sent=self._bytes_sent,
            bytes_received=self._bytes_received,
            isolation=self.isolation_level,
        )