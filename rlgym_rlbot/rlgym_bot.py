from __future__ import annotations

import os
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from traceback import print_exc
from typing import Any, Tuple, Dict, Generic, List, Optional, Union, Callable

import numpy as np
from rlbot import flat
from rlbot.interface import (
    RLBOT_SERVER_IP,
    RLBOT_SERVER_PORT,
    MsgHandlingResult,
    SocketRelay,
)
from rlbot.managers import Bot
from rlbot.managers.rendering import Renderer
from rlbot.utils import fill_desired_game_state
from rlbot.utils.logging import DEFAULT_LOGGER, get_logger
from rlgym.api import (
    ActionParser,
    ActionType,
    AgentID,
    DoneCondition,
    ObsBuilder,
    ObsType,
    RewardFunction,
    RewardType,
    SharedInfoProvider,
    StateMutator,
)
from rlgym_compat import GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from .util import create_base_state


class MissedActionRecoveryStyle(Enum):
    RESET = 0
    IGNORE = 1


class MissedStepTickRecoveryStyle(Enum):
    RESET = 0
    USE_NEAREST = 1


class StepOffsetRelativeTo(Enum):
    ACTION_START = 0
    ACTION_END = 1


@dataclass
class StepOffset:
    offset: int = -1
    relative_to: StepOffsetRelativeTo = StepOffsetRelativeTo.ACTION_END


@dataclass
class RLGymBotConfig(Generic[AgentID]):
    step_offset: StepOffset = field(default_factory=lambda: StepOffset())
    standard_map: bool = True
    sim_extra_info: bool = False
    agent_ids_fn: Optional[Callable[[flat.GamePacket], Dict[int, AgentID]]] = None


class RLGymBot(Generic[AgentID, ActionType, ObsType, RewardType]):
    """
    A convenience base class for bots developed using RLGym.
    The base class handles the setup and communication with the rlbot server, along with management of RLGym config objects.
    Subclass from this to override the following methods:
    - initialize
    - retire
    - handle_reward
    - get_action
    - handle_match_comm

    """

    logger = DEFAULT_LOGGER

    team: int = -1
    index: int = -1
    name: str = ""
    player_id: int = 0

    match_config = flat.MatchConfiguration()
    """
    Contains info about what map you're on, game mode, mutators, etc.
    """

    field_info = flat.FieldInfo()
    """
    Contains info about the map, such as the locations of boost pads and goals.
    """

    ball_prediction = flat.BallPrediction()
    """
    A simulated prediction of the ball's trajectory including collisions with field geometry (but not cars).
    """

    _initialized_bot = False
    _has_match_settings = False
    _has_field_info = False
    _has_player_mapping = False

    _latest_prediction = flat.BallPrediction()

    def __init__(
        self,
        obs_builder: ObsBuilder[AgentID, ObsType, GameState, Any],
        action_parser: ActionParser[AgentID, ActionType, np.ndarray, GameState, Any],
        reward_function: Optional[
            RewardFunction[AgentID, GameState, RewardType]
        ] = None,
        termination_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        truncation_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        state_mutator: Optional[StateMutator[GameState]] = None,
        shared_info_provider: Optional[SharedInfoProvider[AgentID, GameState]] = None,
        default_agent_id: Optional[str] = None,
        config: RLGymBotConfig[AgentID] = RLGymBotConfig(),
    ):
        assert (
            config.step_offset.relative_to != StepOffsetRelativeTo.ACTION_END
            or config.step_offset.offset <= 0
        ), "Cannot have a positive offset relative to action end, as this would imply your next action starts more than 0 ticks after your last action ended."
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.state_mutator = state_mutator
        self.shared_info_provider = shared_info_provider
        self.config = config

        self.received_first_packet = False
        self.shared_info = (
            shared_info_provider.create({}) if shared_info_provider is not None else {}
        )
        self._last_packet = None
        self._first_countdown_tick = 0
        self._last_sent_action_tick = 0
        self._last_sent_action = flat.ControllerState()
        self._latest_engine_action_length = 0
        self._last_sent_action_hardcoded = False
        self._got_countdown_action = False
        self._unused_packets: List[flat.GamePacket] = []
        self._hist_game_states_and_packets: Dict[
            int, Tuple[GameState, flat.GamePacket]
        ] = {}
        self._tick_action_map: Dict[int, flat.ControllerState] = {}
        self._action_boundaries: List[int] = []

        agent_id = os.environ.get("RLBOT_AGENT_ID") or default_agent_id
        if agent_id is None:
            self.logger.critical(
                "Environment variable RLBOT_AGENT_ID is not set and no default agent id is passed to "
                "the constructor of the bot. If you are starting your bot manually, please set it "
                "manually, e.g. `RLBOT_AGENT_ID=<agent_id> python yourbot.py`"
            )
            exit(1)

        self._game_interface = SocketRelay(agent_id, logger=self.logger)
        self._game_interface.match_config_handlers.append(self._handle_match_config)
        self._game_interface.field_info_handlers.append(self._handle_field_info)
        self._game_interface.match_comm_handlers.append(
            self._handle_match_communication
        )
        self._game_interface.ball_prediction_handlers.append(
            self._handle_ball_prediction
        )
        self._game_interface.controllable_team_info_handlers.append(
            self._handle_controllable_team_info
        )
        self._game_interface.packet_handlers.append(self._handle_packet)

        self.renderer = Renderer(self._game_interface)

    def _try_initialize(self):
        if (
            self._initialized_bot
            or not self._has_match_settings
            or not self._has_field_info
            or not self._has_player_mapping
        ):
            # Not ready to initialize
            return

        # Search match settings for our name
        for player in self.match_config.player_configurations:
            match player.variety:
                case flat.CustomBot(name):
                    if player.player_id == self.player_id:
                        self.name = name
                        self.logger = get_logger(self.name)
                        break
        else:  # else block runs if break was not hit
            self.logger.warning(
                "Bot with agent id '%s' did not find itself in the match configuration",
                self._game_interface.agent_id,
            )

        self.latest_game_state: GameState[AgentID] = GameState.create_compat_game_state(
            field_info=self.field_info,
            match_configuration=self.match_config,
            standard_map=self.config.standard_map,
            agent_ids_fn=self.config.agent_ids_fn,
        )
        if self.config.sim_extra_info:
            self.sim_extra_info = SimExtraInfo(self.field_info, self.match_config)
        try:
            self.initialize()
        except Exception as e:
            self.logger.critical(
                "Bot %s failed to initialize due the following error: %s", self.name, e
            )
            print_exc()
            exit()

        self._initialized_bot = True
        self.logger.info(
            "Initialized! Make sure to have a capped framerate of 120 or 240 (NOT 360!) for best performance."
        )
        self._game_interface.send_msg(flat.InitComplete())

    def _handle_match_config(self, match_config: flat.MatchConfiguration):
        self.match_config = match_config
        self._has_match_settings = True
        self._try_initialize()

    def _handle_field_info(self, field_info: flat.FieldInfo):
        self.field_info = field_info
        self._has_field_info = True
        self._try_initialize()

    def _handle_controllable_team_info(
        self, player_mappings: flat.ControllableTeamInfo
    ):
        self.team = player_mappings.team
        controllable = player_mappings.controllables[0]
        self.player_id = controllable.identifier
        self.index = controllable.index
        self._has_player_mapping = True

        self._try_initialize()

    def _handle_ball_prediction(self, ball_prediction: flat.BallPrediction):
        self._latest_prediction = ball_prediction

    # Update self._tick_action_map. The map defines actions where the action self._tick_action_map[i] is intended to be taken between packets with frame_num i and i+1
    def _update_tick_action_map(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        obs: Dict[int, ObsType],
        start_tick: int,
    ):
        if self.config.agent_ids_fn is not None:
            agent_id = self.config.agent_ids_fn(packet)[self.player_id]
        else:
            agent_id = self.player_id
        action = self.get_action(obs[agent_id], game_state, packet)
        prev_latest_engine_action_length = self._latest_engine_action_length
        if isinstance(action, List):
            self._latest_engine_action_length = len(action)
            controller_states = action
        else:
            engine_action = self.action_parser.parse_actions(
                {agent_id: action}, game_state, self.shared_info
            )[agent_id]
            steps = engine_action.shape[0]
            self._latest_engine_action_length = steps
            controller_states = []
            for idx in range(steps):
                (
                    throttle,
                    steer,
                    pitch,
                    yaw,
                    roll,
                    jump_float,
                    boost_float,
                    handbrake_float,
                ) = engine_action[idx]
                controller_states.append(
                    flat.ControllerState(
                        throttle=throttle,
                        steer=steer,
                        pitch=pitch,
                        yaw=yaw,
                        roll=roll,
                        jump=jump_float > 0,
                        boost=boost_float > 0,
                        handbrake=handbrake_float > 0,
                        use_item=False,
                    )
                )

        self._action_boundaries.append(start_tick)
        for idx, controller_state in enumerate(controller_states):
            self._tick_action_map[start_tick + idx] = controller_state

        # Lower bound based on worst case analysis (each action is one tick), minus (prev_latest_engine_action_length + 1) in the case of the first step when you have offset 0 relative to action start
        min_tick_keep = (
            start_tick
            - prev_latest_engine_action_length
            - 1
            + min(0, self.config.step_offset.offset)
        )
        self._tick_action_map = {
            k: v for (k, v) in self._tick_action_map.items() if k >= min_tick_keep
        }
        self._action_boundaries = [
            v for v in self._action_boundaries if v >= min_tick_keep
        ]

    @staticmethod
    def _get_agents_list(game_state: GameState):
        return list(game_state.cars.keys())

    def _update_gamestate_ball_touches(self, reset_tick: int):
        if reset_tick not in self._hist_game_states_and_packets:
            self.logger.warning(
                f"Ball touches reset failed - reset requested starting from tick {reset_tick} but there is no game state that was created on this tick and so we can't identify the number of touches to subtract for each player"
            )
            return

        gs, _ = self._hist_game_states_and_packets[reset_tick]
        n_touches_to_sub = {car_id: car.ball_touches for car_id, car in gs.cars.items()}
        for gs, _ in self._hist_game_states_and_packets.values():
            if gs.tick_count < reset_tick:
                continue
            for car_id, car in gs.cars.items():
                car.ball_touches -= n_touches_to_sub[car_id]
                car._last_reset_ball_touches_tick = reset_tick

    def _env_reset(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: int,
        set_state=False,
    ):
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.create(self.shared_info)
        if self.state_mutator is not None:
            # Whether or not we are going to set the state, we should still make this call so that shared_info can get updated by the state mutator (I guess)
            desired_state = create_base_state()
            self.state_mutator.apply(desired_state, self.shared_info)
            if set_state:
                # TODO: set state
                pass
        game_state = deepcopy(game_state)
        self._update_gamestate_ball_touches(game_state.tick_count)
        agents = RLGymBot._get_agents_list(game_state)
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.set_state(
                agents, game_state, self.shared_info
            )
        self.obs_builder.reset(agents, game_state, self.shared_info)
        self.action_parser.reset(agents, game_state, self.shared_info)
        if self.termination_condition is not None:
            self.termination_condition.reset(agents, game_state, self.shared_info)
        if self.truncation_condition is not None:
            self.truncation_condition.reset(agents, game_state, self.shared_info)
        if self.reward_function is not None:
            self.reward_function.reset(agents, game_state, self.shared_info)
        obs = self.obs_builder.build_obs(agents, game_state, self.shared_info)
        self._update_tick_action_map(packet, game_state, obs, start_tick)

    def _env_step(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: int,
    ) -> Tuple[bool, bool]:
        game_state = deepcopy(game_state)
        self._update_gamestate_ball_touches(game_state.tick_count)
        agents = RLGymBot._get_agents_list(game_state)
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.step(
                agents, game_state, self.shared_info
            )
        obs = self.obs_builder.build_obs(agents, game_state, self.shared_info)
        is_terminated = {self.player_id: False}
        is_truncated = {self.player_id: False}
        if self.termination_condition is not None:
            is_terminated = self.termination_condition.is_done(
                agents, game_state, self.shared_info
            )
        if self.truncation_condition is not None:
            is_truncated = self.truncation_condition.is_done(
                agents, game_state, self.shared_info
            )
        if self.reward_function is not None:
            rewards = self.reward_function.get_rewards(
                agents,
                game_state,
                is_terminated,
                is_truncated,
                self.shared_info,
            )
            self.handle_reward(rewards[self.player_id])
        self._update_tick_action_map(
            packet,
            game_state,
            obs,
            start_tick,
        )
        return is_terminated, is_truncated

    def _handle_packet(self, packet: flat.GamePacket):
        self._unused_packets.append(packet)

    def _update_gamestate_using_unused_packets(self):
        # We assume the packets here are only going to be match phase of countdown, kickoff, or active
        for packet in self._unused_packets:
            extra_info = None
            if self.config.sim_extra_info:
                extra_info = self.sim_extra_info.get_extra_info(packet)
            self.latest_game_state.update(packet, extra_info)
            self._hist_game_states_and_packets[packet.match_info.frame_num] = (
                deepcopy(self.latest_game_state),
                packet,
            )
        cur_tick = self._unused_packets[-1].match_info.frame_num
        min_needed_tick = 0
        min_needed_tick_reset = (
            cur_tick
            + (self.config.step_offset.relative_to == StepOffsetRelativeTo.ACTION_END)
            * self.config.step_offset.offset
        )
        min_needed_tick_step = (
            max(self._tick_action_map, default=0)
            + 1
            + self.config.step_offset.offset
            - (self.config.step_offset.relative_to == StepOffsetRelativeTo.ACTION_START)
            * self._latest_engine_action_length
        )
        min_needed_tick = min(cur_tick, min_needed_tick_reset, min_needed_tick_step)
        self._hist_game_states_and_packets = {
            k: v
            for (k, v) in self._hist_game_states_and_packets.items()
            if k >= min_needed_tick
        }

    # We last sent an action on some tick, and more than one tick has passed since then. That's ok if the action we sent is the one we were going to send anyway for the ticks since.
    # This method finds the ticks where we have taken the wrong actions
    def _get_wrong_tick_action_map(
        self,
    ) -> Dict[int, Tuple[flat.ControllerState, Optional[flat.ControllerState]]]:
        cur_tick = self._unused_packets[-1].match_info.frame_num
        wrong_tick_action_map = {}
        last_sent_action_pack = self._last_sent_action.pack()
        idx = self._last_sent_action_tick
        while idx < cur_tick:
            idx += 1
            if idx in self._tick_action_map:
                desired_action_pack = self._tick_action_map[idx].pack()
                if desired_action_pack != last_sent_action_pack:
                    wrong_tick_action_map[idx] = (
                        flat.ControllerState.unpack(last_sent_action_pack),
                        flat.ControllerState.unpack(desired_action_pack),
                    )
            else:
                wrong_tick_action_map[idx] = (
                    flat.ControllerState.unpack(last_sent_action_pack),
                    None,
                )
        return wrong_tick_action_map

    # Handle env resets when they need to happen but we don't necessarily have all the information to do it normally.
    def _unexpected_env_reset_and_update_action_map(
        self, desired_reset_tick, desired_submit_first_action_tick
    ):
        cur_tick = self._unused_packets[-1].match_info.frame_num
        nearest_hist_tick_to_desired_reset_tick = min(
            self._hist_game_states_and_packets,
            key=lambda v: abs(v - desired_reset_tick),
        )
        (nearest_game_state, nearest_packet) = self._hist_game_states_and_packets[
            nearest_hist_tick_to_desired_reset_tick
        ]

        self._env_reset_and_maybe_step(
            packet=nearest_packet,
            game_state=nearest_game_state,
            start_tick=desired_submit_first_action_tick,
        )
        if cur_tick not in self._tick_action_map:
            # If the action is only for one tick and not self.config.above_240_fps_mode, it will put it for the previous tick instead of the current tick, which is not workable.
            # If it's more than one tick, starting using the second tick is desirable
            self._tick_action_map[cur_tick] = self._tick_action_map[
                desired_submit_first_action_tick
            ]

    def _env_reset_and_maybe_step(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: int,
    ):
        self._env_reset(packet, game_state, start_tick, set_state=False)
        if (
            self.config.step_offset.offset == 0
            and self.config.step_offset.relative_to == StepOffsetRelativeTo.ACTION_START
        ):
            # We calculate the update to the obs etc using the same state as the reset and use this to produce the actions to be used after the reset's obs' actions
            self._env_step(packet, game_state, max(self._tick_action_map.keys()) + 1)

    def _handle_possible_hardcoded_action(self) -> bool:
        latest_packet = self._unused_packets[-1]
        cur_tick = latest_packet.match_info.frame_num
        action = self.get_hardcoded_action(
            self.latest_game_state,
            latest_packet,
            max(
                max(self._tick_action_map, default=0) - cur_tick,
                0,
            ),
        )
        if action is not None:
            # Take hard coded action
            self._last_sent_action_hardcoded = True
            self._tick_action_map.clear()
            self._tick_action_map[self._unused_packets[-1].match_info.frame_num] = (
                action
            )
        else:
            if self._last_sent_action_hardcoded:
                # Reset env on current tick
                self._tick_action_map.clear()
                game_state, packet = self._hist_game_states_and_packets[cur_tick]
                self._env_reset_and_maybe_step(game_state, packet, cur_tick)
            self._last_sent_action_hardcoded = False
        return self._last_sent_action_hardcoded

    # Update the tick action map and internal state based on the unused packets we have collected since the last time this method was called
    def _process_unused_packets(self):
        try:
            # Set state variables that update on the first countdown packet
            if (
                self._last_packet is None
                or self._last_packet.match_info.match_phase
                in [
                    flat.MatchPhase.GoalScored,
                    flat.MatchPhase.Replay,
                ]
            ) and any(
                [
                    p.match_info.match_phase == flat.MatchPhase.Countdown
                    for p in self._unused_packets
                ]
            ):
                self._tick_action_map.clear()
                self._first_countdown_tick = next(
                    p.match_info.frame_num
                    for p in self._unused_packets
                    if p.match_info.match_phase == flat.MatchPhase.Countdown
                )
                self._got_countdown_action = False

            latest_packet = self._unused_packets[-1]
            cur_tick = latest_packet.match_info.frame_num

            # If the latest packet is not one of Countdown, Kickoff, or Active, delegate to self.get_other_packet_output
            if latest_packet.match_info.match_phase not in [
                flat.MatchPhase.Countdown,
                flat.MatchPhase.Kickoff,
                flat.MatchPhase.Active,
            ]:
                self._tick_action_map[cur_tick] = self.get_other_packet_output(
                    latest_packet
                )
                return

            self._update_gamestate_using_unused_packets()

            if latest_packet.match_info.match_phase == flat.MatchPhase.Countdown:
                if cur_tick - self._first_countdown_tick < 440:
                    self._tick_action_map[cur_tick] = self.get_other_packet_output(
                        latest_packet
                    )
                else:
                    if not self._got_countdown_action:
                        used_hardcoded_action = self._handle_possible_hardcoded_action()
                        if not used_hardcoded_action:
                            self._tick_action_map.clear()
                            self._env_reset_and_maybe_step(
                                packet=latest_packet,
                                game_state=self.latest_game_state,
                                start_tick=cur_tick,
                            )
                        self._got_countdown_action = True
                    else:
                        # We are still in countdown but already updated the tick action map. Just move everything in the tick action map forward to start on this tick
                        min_tick = min(self._tick_action_map)
                        self._tick_action_map = {
                            (k + cur_tick - min_tick): v
                            for (k, v) in self._tick_action_map.items()
                        }
                return

            # We might need to reset if we detect that we have done an incorrect action for any tick we are processing
            if (
                len(self._unused_packets) > 1
                or self._unused_packets[0].match_info.frame_num
                > self._last_packet.match_info.frame_num + 1
            ):
                wrong_tick_action_map = self._get_wrong_tick_action_map()
                if len(wrong_tick_action_map) > 0:
                    self.logger.warning("Took incorrect action on a previous tick")
                    (cur_gs, cur_packet) = self._hist_game_states_and_packets[cur_tick]
                    next_action_start_tick = min(
                        v for v in self._action_boundaries if v >= cur_tick
                    )
                    match self.decide_missed_action_recovery_style(
                        cur_gs,
                        cur_packet,
                        next_action_start_tick - cur_tick,
                        wrong_tick_action_map,
                    ):
                        case MissedActionRecoveryStyle.RESET:
                            self._unexpected_env_reset_and_update_action_map(
                                cur_tick, cur_tick
                            )
                            return
                        case MissedActionRecoveryStyle.IGNORE:
                            # Whatever I guess
                            pass

            # Do the below loop while next_env_step_tick <= cur_tick
            should_continue = True
            while should_continue:
                last_defined_action_tick = max(self._tick_action_map)
                next_env_step_tick = (
                    last_defined_action_tick
                    + 1
                    + self.config.step_offset.offset
                    - (
                        self.config.step_offset.relative_to
                        == StepOffsetRelativeTo.ACTION_START
                    )
                    * self._latest_engine_action_length
                )
                if next_env_step_tick < cur_tick:
                    # We were supposed to step the env in the past, which might be fine if we still have actions queued up or the action we've been taking is the intended one
                    # First, find the game state we should use to step
                    step_tick = None
                    if next_env_step_tick not in self._hist_game_states_and_packets:
                        # We missed the tick we wanted to use in order to perform the step
                        self.logger.warning(
                            "missed step tick %s, recovering using missed step tick recovery style",
                            next_env_step_tick,
                        )
                        (cur_gs, cur_packet) = self._hist_game_states_and_packets[
                            cur_tick
                        ]
                        next_action_start_tick = min(
                            v for v in self._action_boundaries if v >= cur_tick
                        )
                        match self.decide_missed_step_tick_recovery_style(
                            cur_gs,
                            cur_packet,
                            next_action_start_tick - cur_tick,
                            self._last_sent_action,
                        ):
                            case MissedStepTickRecoveryStyle.RESET:
                                self._unexpected_env_reset_and_update_action_map(
                                    cur_tick, cur_tick
                                )
                                continue
                            case MissedStepTickRecoveryStyle.USE_NEAREST:
                                step_tick = min(
                                    self._hist_game_states_and_packets,
                                    key=lambda v: abs(v - next_env_step_tick),
                                )
                    else:
                        step_tick = next_env_step_tick
                    step_gs, step_packet = self._hist_game_states_and_packets[step_tick]

                    # Next see if we are still fine and handle the various scenarios
                    if last_defined_action_tick >= cur_tick:
                        # OK, we can still catch up
                        self._env_step(
                            step_packet,
                            step_gs,
                            last_defined_action_tick + 1,
                        )
                    else:
                        # OK, we still might be fine if the action we have been taking is the intended one, so let's try to step and then see if what we planned aligns with what we actually did
                        self._env_step(
                            step_packet, step_gs, last_defined_action_tick + 1
                        )
                        wrong_tick_action_map = self._get_wrong_tick_action_map()
                        if len(wrong_tick_action_map) > 0:
                            self.logger.warning(
                                "missed action, recovering using missed action recovery style"
                            )
                            (cur_gs, cur_packet) = self._hist_game_states_and_packets[
                                cur_tick
                            ]
                            next_action_start_tick = min(
                                v for v in self._action_boundaries if v >= cur_tick
                            )
                            match self.decide_missed_action_recovery_style(
                                cur_gs,
                                cur_packet,
                                next_action_start_tick - cur_tick,
                                wrong_tick_action_map,
                            ):
                                case MissedActionRecoveryStyle.RESET:
                                    self._unexpected_env_reset_and_update_action_map(
                                        cur_tick, cur_tick
                                    )
                                    return
                                case MissedActionRecoveryStyle.IGNORE:
                                    # Whatever I guess
                                    pass
                else:
                    if next_env_step_tick == cur_tick:
                        self._env_step(
                            latest_packet,
                            self.latest_game_state,
                            last_defined_action_tick + 1,
                        )
                    else:
                        should_continue = False

        except Exception as e:
            self.logger.error(
                "Bot %s encountered an error while processing game packet: %s",
                self.name,
                e,
            )
            print_exc()

    def _run(self):
        running = True
        block_next = False

        while running:
            # If there might be more messages, check for another one with blocking=False
            # if there are no more messages, process the latest packet then wait for the next message with blocking=True
            result = self._game_interface.handle_incoming_messages(blocking=block_next)
            block_next = False
            match result:
                case MsgHandlingResult.TERMINATED:
                    running = False
                case MsgHandlingResult.NO_INCOMING_MSGS:
                    if len(self._unused_packets) > 0:
                        self._process_unused_packets()
                        self._last_packet = self._unused_packets[-1]
                        self._unused_packets.clear()
                        cur_tick = self._last_packet.match_info.frame_num
                        if cur_tick in self._tick_action_map:
                            self._last_sent_action_tick = cur_tick
                            self._last_sent_action = self._tick_action_map[cur_tick]
                        else:
                            if self._last_packet.match_info.match_phase in [
                                flat.MatchPhase.Countdown,
                                flat.MatchPhase.Kickoff,
                                flat.MatchPhase.Active,
                            ]:
                                self.logger.warning(
                                    "No controller state set in self._tick_action_map for this tick (%d)!",
                                    cur_tick,
                                )
                        self._game_interface.send_msg(
                            flat.PlayerInput(self.index, self._last_sent_action)
                        )
                    block_next = True
                case _:
                    pass

    def run(
        self,
        *,
        wants_match_communications: bool = True,
        wants_ball_predictions: bool = True,
    ):
        """
        Runs the bot. This operation is blocking until the match ends.
        """

        rlbot_server_ip = os.environ.get("RLBOT_SERVER_IP", RLBOT_SERVER_IP)
        rlbot_server_port = int(os.environ.get("RLBOT_SERVER_PORT", RLBOT_SERVER_PORT))

        try:
            self._game_interface.connect(
                wants_match_communications=wants_match_communications,
                wants_ball_predictions=wants_ball_predictions,
                rlbot_server_ip=rlbot_server_ip,
                rlbot_server_port=rlbot_server_port,
            )

            self._run()
        except Exception as e:
            self.logger.error("Unexpected error: %s", e)
            print_exc()
        finally:
            self.retire()
            del self._game_interface

    def _handle_match_communication(self, match_comm: flat.MatchComm):
        self.handle_match_comm(
            match_comm.index,
            match_comm.team,
            match_comm.content,
            match_comm.display,
            match_comm.team_only,
        )

    def handle_match_comm(
        self,
        index: int,
        team: int,
        content: bytes,
        display: Optional[str],
        team_only: bool,
    ):
        """
        Called when a match communication message is received.
        See `send_match_comm`.
        NOTE: Messages from scripts will have `team == 2` and the index will be its index in the match settings.
        """

    def send_match_comm(
        self, content: bytes, display: Optional[str] = None, team_only: bool = False
    ):
        """
        Emits a match communication message to other bots and scripts.

        - `content`: The content of the message containing arbitrary data.
        - `display`: The message to be displayed in the game in "quick chat", or `None` to display nothing.
        - `team_only`: If True, only your team will receive the message.
        """
        self._game_interface.send_match_comm(
            flat.MatchComm(
                self.index,
                self.team,
                team_only,
                display,
                content,
            )
        )

    def set_game_state(
        self,
        balls: dict[int, flat.DesiredBallState] = {},
        cars: dict[int, flat.DesiredCarState] = {},
        match_info: Optional[flat.DesiredMatchInfo] = None,
        commands: list[str] = [],
    ):
        """
        Sets the game to the desired state.
        Through this it is possible to manipulate the position, velocity, and rotations of cars and balls, and more.
        See wiki for a full break down and examples.
        """

        game_state = fill_desired_game_state(balls, cars, match_info, commands)
        self._game_interface.send_game_state(game_state)

    def set_loadout(self, loadout: flat.PlayerLoadout, index: Optional[int] = None):
        """
        Sets the loadout of a bot.
        Can be used to select or generate a loadout for the match when called inside `initialize`.
        Does nothing if called outside `initialize` unless state setting is enabled in which case it
        respawns the car with the new loadout.
        """
        self._game_interface.send_set_loadout(
            flat.SetLoadout(index or self.index, loadout)
        )

    def initialize(self):
        """
        Called when the bot is ready for initialization. Field info, match settings, name, index, and team are
        fully loaded at this point, and will not return garbage data unlike in `__init__`.
        """

    def retire(self):
        """Called when the bot is shut down"""

    def handle_reward(self, reward: RewardType):
        """
        Called if reward_function is not None each time a step of the simulated RLGym environment happens
        """
        pass

    def get_other_packet_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        """Allows for cosmetic action input during phases where actions taken have only visual effects.

        Called any time the latest packet received has a match phase not in Countdown, Kickoff, or Active. Also called for the first 460 ticks of Countdown, because Countdown always lasts at least 479 ticks.
        May also be used as a point in execution to send match communications including chat messages.

        Args:
            packet: The latest flat.GamePacket received.
        Returns:
            A ControllerState to use for this packet, mostly (entirely?) relevant during Countdown.
        """
        return flat.ControllerState()

    @abstractmethod
    def get_action(
        self, obs: ObsType, game_state: GameState, packet: flat.GamePacket
    ) -> ActionType:
        """Defines how an ObsType is turned into an ActionType for your bot.

        Called every time the RLGym v2 env simulation creates a new observation and needs an action for that observation.

        Args:
            obs: The ObsType that normally appears in a Dict[AgentID, ObsType] in the obs_dict returned by a RLGym v2 env's step method for this particular agent.
            game_state: The GameState that was fed to the ObsBuilder to get the obs.
            packet: The latest GamePacket that was used to construct the game_state.
        Returns:
            An ActionType to be fed into the ActionParser (as a Dict[AgentID, ActionType]) to get the actions to use in game.
        """
        raise NotImplementedError

    def get_hardcoded_action(
        self, game_state: GameState, packet: flat.GamePacket, ticks_left_in_action: int
    ) -> Optional[flat.ControllerState]:
        """Allows for injection of a hard-coded action.

        Called every Kickoff or Active tick to allow the user to override the pure RLGym v2 env simulation's proceedings in terms of action generation in order to submit a hard-coded action.
        Also called for exactly one Countdown tick near the end of the Countdown match phase (see Returns below)

        Args:
            game_state: The GameState calculated from the latest flat.GamePacket received.
            packet: The latest flat.GamePacket received.
            ticks_left_in_action: The number of ticks (>= 0) left in the sequence of EngineActions returned by the most recent call to the
                ActionParser instance, or 0 if there has not been a call to the ActionParser since the simulated RLGym v2 env's last reset call, or if this method returned a non-None value last tick.
        Returns:
            An optional ControllerState to use on the next tick, or, when called during the Countdown match phase, to use between the last tick of Countdown and the first tick of Kickoff.
            If None, the RLGym v2 env simulation proceeds as normal. If non-None, a reset of the simulated RLGym v2 env will be performed the next time this method returns None.
        """
        return None

    def decide_missed_action_recovery_style(
        self,
        game_state: GameState,
        packet: flat.GamePacket,
        ticks_left_in_action: int,
        wrong_tick_action_map: Dict[
            int, Tuple[flat.ControllerState, Optional[flat.ControllerState]]
        ],
    ) -> MissedActionRecoveryStyle:
        """Determines how detected missed actions (the wrong action was taken for a tick) are handled."""
        # TODO: finish documentation and make implementation more clever (are we in the middle of an action? A flip?)
        return MissedActionRecoveryStyle.IGNORE

    def decide_missed_step_tick_recovery_style(
        self,
        game_state: GameState,
        packet: flat.GamePacket,
        ticks_left_in_action: int,
        last_action: flat.ControllerState,
    ) -> MissedActionRecoveryStyle:
        """Determines how a detected missed step tick (the tick needed for the game state to be used for the calls to the config objects for this step) are handled."""
        # TODO: finish documentation and make implementation more clever (are we in the middle of an action? A flip?)
        return MissedStepTickRecoveryStyle.USE_NEAREST
