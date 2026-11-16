# wumpus_sdk/codec.py
from __future__ import annotations

import json
from wumpus_sdk.dto import (
    ActionDTO,
    AgentErrorDTO,
    GameOverDTO,
    HelloDTO,
    InvalidActionDTO,
    ObservationDTO,
    SessionInitDTO,
    Outcome,
    WireMessage,
)
from wumpus_sdk.errors import ProtocolError
from wumpus_sdk.percept import Percept
from wumpus_sdk.types import Direction, Position


def dto_to_json(dto: WireMessage) -> str:
    """Serialize any SDK DTO to a JSON wire string with a discriminator type field."""
    if isinstance(dto, HelloDTO):
        data = {
            "type": "HelloDTO",
            "protocol_version": dto.protocol_version,
            "sdk_version": dto.sdk_version,
            "agent_name": dto.agent_name,
        }
    elif isinstance(dto, SessionInitDTO):
        data = {
            "type": "SessionInitDTO",
            "protocol_version": dto.protocol_version,
            "grid_width": dto.grid_width,
            "grid_height": dto.grid_height,
            "start_position": dto.start_position.to_wire(),
            "start_direction": dto.start_direction.value,
            "initial_arrows": dto.initial_arrows,
            "max_actions": dto.max_actions,
            "action_timeout_s": dto.action_timeout_s,
            "score_rules": dto.score_rules,
            "agent_seed": dto.agent_seed,
        }
    elif isinstance(dto, ObservationDTO):
        data = {
            "type": "ObservationDTO",
            "step_index": dto.step_index,
            "position": dto.position.to_wire(),
            "direction": dto.direction.value,
            "arrows_remaining": dto.arrows_remaining,
            "has_gold": dto.has_gold,
            "percept": dto.percept.to_wire(),
        }
    elif isinstance(dto, ActionDTO):
        data = {
            "type": "ActionDTO",
            "step_index": dto.step_index,
            "action": dto.action,
        }
    elif isinstance(dto, InvalidActionDTO):
        data = {
            "type": "InvalidActionDTO",
            "step_index": dto.step_index,
            "reason": dto.reason,
            "strikes_used": dto.strikes_used,
            "strikes_allowed": dto.strikes_allowed,
        }
    elif isinstance(dto, GameOverDTO):
        data = {
            "type": "GameOverDTO",
            "outcome": dto.outcome.value,
            "final_score": dto.final_score,
            "actions_taken": dto.actions_taken,
        }
    elif isinstance(dto, AgentErrorDTO):
        data = {
            "type": "AgentErrorDTO",
            "step_index": dto.step_index,
            "error_type": dto.error_type,
            "message": dto.message,
            "traceback": dto.traceback,
        }
    else:
        raise ValueError(f"Unknown DTO type: {type(dto)}")

    return json.dumps(data)


def dto_from_json(raw_json: str) -> WireMessage:
    """Deserialize a JSON wire string into its corresponding DTO object."""
    try:
        data = json.loads(raw_json)
    except Exception as e:
        raise ProtocolError(f"Invalid JSON payload: {e}") from e

    if not isinstance(data, dict) or "type" not in data:
        raise ProtocolError("DTO JSON must be an object with a 'type' field")

    msg_type = data["type"]
    try:
        if msg_type == "HelloDTO":
            return HelloDTO(
                protocol_version=int(data["protocol_version"]),
                sdk_version=str(data["sdk_version"]),
                agent_name=str(data["agent_name"]),
            )
        if msg_type == "SessionInitDTO":
            return SessionInitDTO(
                protocol_version=int(data["protocol_version"]),
                grid_width=int(data["grid_width"]),
                grid_height=int(data["grid_height"]),
                start_position=Position.from_wire(data["start_position"]),
                start_direction=Direction(data["start_direction"]),
                initial_arrows=int(data["initial_arrows"]),
                max_actions=int(data["max_actions"]),
                action_timeout_s=float(data["action_timeout_s"]),
                score_rules={str(k): int(v) for k, v in dict(data["score_rules"]).items()},
                agent_seed=int(data["agent_seed"]),
            )
        if msg_type == "ObservationDTO":
            return ObservationDTO(
                step_index=int(data["step_index"]),
                position=Position.from_wire(data["position"]),
                direction=Direction(data["direction"]),
                arrows_remaining=int(data["arrows_remaining"]),
                has_gold=bool(data["has_gold"]),
                percept=Percept.from_wire(data["percept"]),
            )
        if msg_type == "ActionDTO":
            return ActionDTO(
                step_index=int(data["step_index"]),
                action=str(data["action"]),
            )
        if msg_type == "InvalidActionDTO":
            return InvalidActionDTO(
                step_index=int(data["step_index"]),
                reason=str(data["reason"]),
                strikes_used=int(data["strikes_used"]),
                strikes_allowed=int(data["strikes_allowed"]),
            )
        if msg_type == "GameOverDTO":
            return GameOverDTO(
                outcome=Outcome(data["outcome"]),
                final_score=int(data["final_score"]),
                actions_taken=int(data["actions_taken"]),
            )
        if msg_type == "AgentErrorDTO":
            return AgentErrorDTO(
                step_index=int(data["step_index"]),
                error_type=str(data["error_type"]),
                message=str(data["message"]),
                traceback=str(data["traceback"]),
            )
        raise ProtocolError(f"Unknown DTO type string: {msg_type!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Malformed {msg_type} payload: {exc}") from exc