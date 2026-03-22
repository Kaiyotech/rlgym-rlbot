from dataclasses import dataclass


@dataclass(eq=True, frozen=True, slots=True)
class Id:
    agent_id: str
    player_name: str
    spawn_id: int
