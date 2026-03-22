import numpy as np
from rlbot.flat import FieldInfo
from rlgym_compat import GameConfig, GameState, PhysicsObject


def create_base_state(field_info: FieldInfo):
    gs = GameState()
    gs.tick_count = 0
    gs.goal_scored = False

    gs.config = GameConfig()
    gs.config.gravity = 1
    gs.config.boost_consumption = 1
    gs.config.dodge_deadzone = 0.5

    gs.ball = PhysicsObject()
    gs.cars = {}
    gs.boost_pad_timers = np.zeros(len(field_info.boost_pads), dtype=np.float32)

    return gs
