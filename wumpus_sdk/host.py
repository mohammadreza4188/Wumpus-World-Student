# wumpus_sdk/host.py
from __future__ import annotations

import importlib
import sys
import traceback
from wumpus_sdk.actions import Action
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


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m wumpus_sdk.host <entry_point>", file=sys.stderr)
        sys.exit(1)

    entry_point = sys.argv[1]
    try:
        mod_name, class_name = entry_point.split(":", 1)
        module = importlib.import_module(mod_name)
        agent_cls = getattr(module, class_name)
        agent = agent_cls()
    except Exception as exc:
        err_dto = AgentErrorDTO(
            step_index=-1,
            error_type=type(exc).__name__,
            message=f"Failed to load agent {entry_point}: {exc}",
            traceback=traceback.format_exc(),
        )
        sys.stdout.write(dto_to_json(err_dto) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Send HelloDTO handshake
    hello = HelloDTO(
        protocol_version=1,
        sdk_version="0.1.0",
        agent_name=agent.__class__.__name__,
    )
    sys.stdout.write(dto_to_json(hello) + "\n")
    sys.stdout.flush()

    last_obs: ObservationDTO | None = None
    current_step_index = 0

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            dto = dto_from_json(line)
            if isinstance(dto, SessionInitDTO):
                agent._bind_session(dto)
                agent.on_game_start(dto)
            elif isinstance(dto, ObservationDTO):
                last_obs = dto
                current_step_index = dto.step_index
                action = agent.act(dto)
                reply = ActionDTO(step_index=dto.step_index, action=action.value)
                sys.stdout.write(dto_to_json(reply) + "\n")
                sys.stdout.flush()
            elif isinstance(dto, InvalidActionDTO):
                agent.on_invalid_action(dto)
                if last_obs is not None:
                    action = agent.act(last_obs)
                else:
                    action = Action.TURN_RIGHT
                reply = ActionDTO(step_index=dto.step_index, action=action.value)
                sys.stdout.write(dto_to_json(reply) + "\n")
                sys.stdout.flush()
            elif isinstance(dto, GameOverDTO):
                agent.on_game_over(dto)
                sys.exit(0)
        except Exception as exc:
            err_dto = AgentErrorDTO(
                step_index=current_step_index,
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            sys.stdout.write(dto_to_json(err_dto) + "\n")
            sys.stdout.flush()
            sys.exit(1)


if __name__ == "__main__":
    main()