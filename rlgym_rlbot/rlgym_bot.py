from __future__ import annotations

import os
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from traceback import print_exc
from typing import Any, Tuple, Dict, Generic, List, Optional

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

from .rlgym_state_to_rlbot_state import gamestate_rlgym_to_rlbot
from .util import create_base_state

AgentID = int


class MissedActionRecoveryStyle(Enum):
    RESET = 0
    IGNORE = 1


class MissedStepTickRecoveryStyle(Enum):
    RESET = 0
    USE_NEAREST = 1


@dataclass
class RLGymBotConfig:
    action_step_idx_used_to_build_game_state_for_env_step = -2
    above_240_fps_mode = False
    missed_action_recovery_style = MissedActionRecoveryStyle.RESET
    missed_step_tick_recovery_style = MissedStepTickRecoveryStyle.RESET
    standard_map = True
    sim_extra_info = False


class RLGymBot(Generic[ActionType, ObsType, RewardType]):
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
        config=RLGymBotConfig(),
    ):
        # Sort term, we have no above 240 fps mode implementation and no implementation of action_step_idx_used_to_build_game_state_for_env_step == -1, so let's assert that these aren't set.
        assert (
            not config.above_240_fps_mode
        ), "above_240_fps_mode is not implemented yet!"
        assert (
            config.action_step_idx_used_to_build_game_state_for_env_step != -1
        ), "action_step_idx_used_to_build_game_state_for_env_step == -1 is not implemented yet!"
        # Long term, the below assertion is what we will use. It doesn't hurt to leave it uncommented along with the above.
        assert (
            config.action_step_idx_used_to_build_game_state_for_env_step != -1
            or config.above_240_fps_mode
        ), "action_step_idx_used_to_build_game_state_for_env_step of -1 requires a stable fps of at least 240"
        if config.above_240_fps_mode:
            self.logger.info(
                "above_240_fps_mode enabled. Make sure to have a consistent FPS above 240 in Rocket League for best performance of the bot."
            )
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
        self._just_handled_countdown_packet = False
        self._match_first_countdown = True
        self._last_packet = None
        self._first_countdown_tick = 0
        self._last_env_action_start_tick = 0
        self._last_sent_action_tick = 0
        self._unused_packets: List[flat.GamePacket] = []
        self._hist_game_states_and_packets: Dict[
            int, Tuple[GameState, flat.GamePacket]
        ] = {}
        self._future_tick_action_map: Dict[int, flat.ControllerState] = {}

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

        self.latest_game_state = GameState.create_compat_game_state(
            field_info=self.field_info,
            match_configuration=self.match_config,
            standard_map=self.config.standard_map,
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

    # Update self._future_tick_action_map. The map defines actions where the action self._future_tick_action_map[i] is taken between packets with
    # frame_num i+1 and i+2 if not self.config.above_240_fps_mode and i and i+1 otherwise, i.e., i is when the action is submitted, not when the action is used
    def _update_future_tick_action_map(
        self, packet: flat.GamePacket, obs: Dict[int, ObsType], start_tick: int
    ):
        self._future_tick_action_map = {
            tick: action
            for (tick, action) in self._future_tick_action_map.items()
            if tick >= self._last_env_action_start_tick
        }
        action = self.get_action(obs[self.player_id], packet)
        engine_action = self.action_parser.parse_actions(
            {self.player_id: action}, self.latest_game_state, self.shared_info
        )[self.player_id]
        steps = engine_action.shape[0]
        self._last_engine_action_length = steps
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
            self._future_tick_action_map[start_tick + idx] = flat.ControllerState(
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

    @staticmethod
    def _get_agents_list(game_state: GameState):
        # TODO: maybe agents should only be this agent
        return list(game_state.cars.keys())

    def _update_game_state_ball_touches(self, reset_tick: int):
        assert (
            reset_tick in self._hist_game_states_and_packets
        ), f"Ball touches reset failed - reset requested starting from tick {reset_tick} but there is no game state that was created on this tick and so we can't identify the number of touches to subtract for each player"

        gs, _ = self._hist_game_states_and_packets[reset_tick]
        n_touches_to_sub = {car_id: car.ball_touches for car_id, car in gs.cars.items()}
        for gs, _ in self._hist_game_states_and_packets.values():
            if gs.tick_count < reset_tick:
                continue
            for car_id, car in gs.cars.items():
                car.ball_touches -= n_touches_to_sub[car_id]
                car._last_reset_ball_touches_tick = reset_tick

    def _env_reset(
        self, game_state: GameState, start_tick: int, set_state=False
    ) -> Dict[AgentID, ObsType]:

        self._last_env_action_start_tick = start_tick
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
        self._update_game_state_ball_touches(start_tick)
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
        return self.obs_builder.build_obs(agents, game_state, self.shared_info)

    def _env_step(self, game_state: GameState, start_tick: int):
        self._last_env_action_start_tick = start_tick
        game_state = deepcopy(game_state)
        self._update_game_state_ball_touches(start_tick)
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
        return obs, is_terminated, is_truncated

    def _handle_packet(self, packet: flat.GamePacket):
        # Other match phases don't have a ball and will break things if we try to update the game state, so we may as well ignore them everywhere
        if packet.match_info.match_phase in [
            flat.MatchPhase.Countdown,
            flat.MatchPhase.Kickoff,
            flat.MatchPhase.Active,
        ]:
            self._unused_packets.append(packet)

    def _update_gamestate_using_packets(self, packets: List[flat.GamePacket]):
        # We assume the packets here are only going to be match phase of countdown, kickoff, or active
        for packet in packets:
            extra_info = None
            if self.config.sim_extra_info:
                extra_info = self.sim_extra_info.get_extra_info(packet)
            self.latest_game_state.update(packet, extra_info)
            self._hist_game_states_and_packets[packet.match_info.frame_num] = (
                deepcopy(self.latest_game_state),
                packet,
            )
        self._hist_game_states_and_packets = {
            tick: gsp
            for tick, gsp in self._hist_game_states_and_packets.items()
            if tick >= self._last_env_action_start_tick
        }

    # We last sent an action on some tick, and more than one tick has passed since then. That's ok if the action we sent is the one we were going to send anyway for the ticks since.
    # This method checks to see if that's indeed the case or not
    def _check_last_sent_action_correct_for_ticks_since(self) -> bool:
        cur_tick = self._unused_packets[-1].match_info.frame_num
        correct = False
        if cur_tick in self._future_tick_action_map:
            # If cur_tick is in self._future_tick_action_map, we only need to reset if the action we have taken since then up to the current tick is not equal to
            # self._future_tick_action_map[self._last_sent_action_tick]
            correct = True
            last_sent_action_pack = self._future_tick_action_map[
                self._last_sent_action_tick
            ].pack()
            idx = self._last_sent_action_tick
            while correct and idx < cur_tick:
                idx += 1
                if self._future_tick_action_map[idx].pack() != last_sent_action_pack:
                    correct = False
        return correct

    def _env_reset_and_update_action_map(
        self, packet: flat.GamePacket, start_tick: bool, set_state: bool = False
    ):
        obs = self._env_reset(self.latest_game_state, start_tick, set_state=False)
        self._update_future_tick_action_map(packet, obs, start_tick)
        if self.config.action_step_idx_used_to_build_game_state_for_env_step == 0:
            # We calculate the update to the obs etc using the same state as the reset and use this to produce the actions to be used after the reset's obs' actions
            (obs, *_) = self._env_step(self.latest_game_state, start_tick)
            self._update_future_tick_action_map(
                packet, obs, max(self._future_tick_action_map.keys()) + 1
            )

    # Update the future tick action map and internal state based on the unused packets we have collected since the last time this method was called
    def _process_unused_packets(self):
        try:
            # Check if we are past the first countdown of the match so that we can know whether or not we can use the 480 tick thing to line up kickoff actions appropriately (see below)
            if any(
                [
                    p.match_info.match_phase
                    in [flat.MatchPhase.Kickoff, flat.MatchPhase.Active]
                    for p in self._unused_packets
                ]
            ):
                self._match_first_countdown = False

            # If we just received a countdown packet for the first time since the last countdown, store the first countdown packet's frame num for use later
            if self._last_packet in [
                None,
                flat.MatchPhase.GoalScored,
                flat.MatchPhase.Replay,
            ] and any(
                [
                    p.match_info.match_phase == flat.MatchPhase.Countdown
                    for p in self._unused_packets
                ]
            ):
                self._first_countdown_tick = next(
                    p.match_info.frame_num
                    for p in self._unused_packets
                    if p.match_info.match_phase == flat.MatchPhase.Countdown
                )

            latest_packet = self._unused_packets[-1]

            # If the latest packet is not one of Countdown, Kickoff, or Active, delegate to self.get_other_packet_output
            if latest_packet.match_info.match_phase not in [
                flat.MatchPhase.Countdown,
                flat.MatchPhase.Kickoff,
                flat.MatchPhase.Active,
            ]:
                self._game_interface.send_msg(
                    flat.PlayerInput(
                        self.index, self.get_other_packet_output(latest_packet)
                    )
                )
                return

            # TODO: update this for self.above_240_fps_mode = True
            # If the latest packet is countdown, update the game state using all the packets sequentially and then handle the latest packet, and then we're done
            if latest_packet.match_info.match_phase == flat.MatchPhase.Countdown:
                cur_tick = latest_packet.match_info.frame_num
                # Actually, if this isn't the first kickoff, we can do a little better by assuming we caught the first tick of countdown and countdown lasts for 480 ticks:
                # In that case, we can predict when we are on the last tick of countdown and when this happens we actually don't want to update the future tick action map - we want to leave the
                # actions in place from the previous packet so that we are aligned properly. Of course, we only want to do this if 240 fps mode is disabled, because in that case we
                # want to use the last tick of countdown to reset and determine actions.
                if (
                    not self.config.above_240_fps_mode
                    and not self._match_first_countdown
                    and cur_tick - self._first_countdown_tick >= 479
                ):
                    # We currently are processing what is expected to be the last countdown packet. We therefore want to reset using the second to last packet,
                    # which we might've already handled on a previous invocation. If we have, just update the game state and return - the future tick action map is
                    # already populated. If not, reset using the second to last packet. This upcoming action (the one which is too late to control when not in
                    # 240 fps mode) will be a previous tick's first action which is fine, but we really want to use the second action now.
                    n_packets = len(self._unused_packets)
                    if n_packets > 1:
                        self._update_gamestate_using_packets(self._unused_packets[:-1])
                        # Setting cur_tick=cur_tick-1 means that the action will have "started" a tick ago from the perspective of the rest of the code, so the action we
                        # take starting now will be the second action as desired. Note that this really is what we desire because we are definitely still in countdown as of the latest packet.
                        self._env_reset_and_update_action_map(
                            packet=self._unused_packets[-2],
                            start_tick=cur_tick - 1,
                            set_state=False,
                        )
                        self._update_gamestate_using_packets([self._unused_packets[-1]])
                    else:
                        self._update_gamestate_using_packets(self._unused_packets)
                    return

                self._update_gamestate_using_packets(self._unused_packets)
                # Get new action based on the latest packet and set future tick action map accordingly. We want to start moving between the last tick of countdown and
                # the first tick of kickoff, before we even know it's kickoff in this code
                self._env_reset_and_update_action_map(
                    packet=self._unused_packets[-1],
                    start_tick=cur_tick,
                    set_state=False,
                )
                self._just_handled_countdown_packet = True
                return
            cur_tick = latest_packet.match_info.frame_num
            if (
                self._unused_packets[0].match_info.match_phase
                == flat.MatchPhase.Countdown
            ):
                # TODO: update this for self.above_240_fps_mode = True
                # We have some positive number of countdown packets and some positive number of live game packets (either kickoff or active, but probably kickoff unless something crazy has happened).
                if not self._check_last_sent_action_correct_for_ticks_since():
                    # On a previous tick, we took an action we did not expect to have taken initially. Let's reset and start fresh, and then we're done.
                    # I'm not going to use MissedActionRecoveryStyle here because this is an edge case where resetting should always be preferable.
                    self._update_gamestate_using_packets(self._unused_packets)
                    self._env_reset_and_update_action_map(
                        packet=self._unused_packets[-1],
                        start_tick=cur_tick,
                        set_state=False,
                    )
                    return
                # OK, at this point we are in the same situation we would be in if all the unused packets were live game ones, so let's handle both at once by finishing this if block here and continuing below

            # We need to update the game state using all the ticks, but we want to grab the correct game state based on the action_step_idx_used_to_build_game_state_for_env_step config value
            # Let's first look at how this value works in RLGym v2, or more specifically how I would imagine it working if it existed.
            # For simplicity, let's say every action (by this I mean result from the action parser) has a shape of (4,8) i.e. it defines 4 ticks worth of env actions.
            # . represents a regular state, and - represents a state transition (due to an env action or a void state),
            # A value of 0 would look like the below - in particular, the (n+1)th action is decided using the state that the nth action is about to start from:
            # .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # || | | ||| | | ||| | | └Action 3 tick 4, etc
            # || | | ||| | | ||| | └Action 3 tick 3
            # || | | ||| | | ||| └Action 3 tick 2
            # || | | ||| | | ||└Action 3 tick 1
            # || | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
            # || | | ||| | | └Action 2 tick 4
            # || | | ||| | └Action 2 tick 3
            # || | | ||| └Action 2 tick 2
            # || | | ||└Action 2 tick 1
            # || | | |└Obs 3 created using this state, action 3 decided using obs 3
            # || | | └Action 1 tick 4
            # || | └Action 1 tick 3
            # || └Action 1 tick 2
            # |└Action 1 tick 1
            # └Reset occurs, obs 1 and 2 created using this state, actions 1 and 2 decided using obs 1 and 2 respectively

            # A value of 1 would look like the below - in particular, an action is decided using the state one tick after the last action's env action ticks began:
            # .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # |||| | | ||| | | ||| | └Action 3 tick 4, etc
            # |||| | | ||| | | ||| └Action 3 tick 3
            # |||| | | ||| | | ||└Action 3 tick 2
            # |||| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
            # |||| | | ||| | | └Action 3 tick 1
            # |||| | | ||| | └Action 2 tick 4
            # |||| | | ||| └Action 2 tick 3
            # |||| | | ||└Action 2 tick 2
            # |||| | | |└Obs 3 created using this state, action 3 decided using obs 3
            # |||| | | └Action 2 tick 1
            # |||| | └Action 1 tick 4
            # |||| └Action 1 tick 3
            # |||└Action 1 tick 2
            # ||└Obs 2 created using this state, action 2 decided using obs 2
            # |└Action 1 tick 1
            # └Reset occurs, obs 1 created using this state, action 1 decided using obs 1

            # A value of -1 would look like the below - in particular, the next action action is decided using the state after the last action's env action ticks are done:
            # .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # || | | ||| | ||| | | ||└Action 3 tick 4, etc
            # || | | ||| | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
            # || | | ||| | ||| | | └Action 3 tick 3
            # || | | ||| | ||| | └Action 3 tick 2
            # || | | ||| | ||| └Action 3 tick 1
            # || | | ||| | ||└Action 2 tick 4
            # || | | ||| | |└Obs 3 created using this state, action 3 decided using obs 3
            # || | | ||| | └Action 2 tick 3
            # || | | ||| └Action 2 tick 2
            # || | | ||└Action 2 tick 1
            # || | | |└Obs 2 created using this state, action 2 decided using obs 2
            # || | | └Action 1 tick 4
            # || | └Action 1 tick 3
            # || └Action 1 tick 2
            # |└Action 1 tick 1
            # └Reset occurs, obs 1 created using this state, action 1 decided using obs 1

            # A value of -2 would look like the below - in particular, an action is decided using the state one tick before the last action's env action ticks are done:
            # .-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # || | | ||| | | ||| | | ||└Action 3 tick 4, etc
            # || | | ||| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
            # || | | ||| | | ||| | | └Action 3 tick 3
            # || | | ||| | | ||| | └Action 3 tick 2
            # || | | ||| | | ||| └Action 3 tick 1
            # || | | ||| | | ||└Action 2 tick 4
            # || | | ||| | | |└Obs 3 created using this state, action 3 decided using obs 3
            # || | | ||| | | └Action 2 tick 3
            # || | | ||| | └Action 2 tick 2
            # || | | ||| └Action 2 tick 1
            # || | | ||└Action 1 tick 4
            # || | | |└Obs 2 created using this state, action 2 decided using obs 2
            # || | | └Action 1 tick 3
            # || | └Action 1 tick 2
            # || └Action 1 tick 1
            # |└"void" state transition where sim is stepped without any cars' controls being set
            # └Reset occurs, obs 1 created using this state, action 1 decided using obs 1

            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
            # Let's investigate how we can match this in RLBot. Matching the -1 case is impossible when running Rocket League at 120fps, but is possible at higher frame rates.
            # We will account for that separately - let's focus on the other cases for now.
            # When running at 120fps, submitting an action when the latest packet is for tick n will mean that action gets used in the tick (n+1) to tick (n+2) transition.
            # We will introduce an additional symbol - c will indicate a countdown state. We will need this because we will want to start before the first tick of kickoff.
            # In particular, the last countdown state is followed by a state transition that actually uses whatever actions have been input.
            # For values above -1, we cannot match RLGym exactly for the reset state because that would require taking an action for a state immediately following seeing that state, which is not possible at 120fps.
            # Instead, we will use the last tick of the countdown. Previously I have written "Action n tick k" but now I will explicitly write that this is the state transition where that
            # gets used, not where it is submitted, because in RLBot that distinction matters. This one is going to look quite cluttered, but it is easiest to do it this way to line it up with the
            # diagrams above. Later we will simplify this down by removing the lines where env actions get used, since only the submitted ones matter for our logic.

            # How can we match the 0 case? In RLGym, the reset state is the state directly preceding the tick that action 1 tick 1 is used. However, we can't generate the obs for this
            # and submit it at the correct time, so we use the state during the countdown before the proper reset state as a stand-in since the cars aren't taking any actions anyway (they are just settling)
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | |||||||||||||||||||||||└Action 3 tick 4 used, etc
            # | ||||||||||||||||||||||└Action 4 tick 1 submitted
            # | |||||||||||||||||||||└Action 3 tick 3 used
            # | ||||||||||||||||||||└Action 3 tick 4 submitted
            # | |||||||||||||||||||└Action 3 tick 2 used
            # | ||||||||||||||||||└Action 3 tick 3 submitted
            # | |||||||||||||||||└Action 3 tick 1 used
            # | ||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 2 submitted
            # | |||||||||||||||└Action 2 tick 4 used
            # | ||||||||||||||└Action 3 tick 1 submitted
            # | |||||||||||||└Action 2 tick 3 used
            # | ||||||||||||└Action 2 tick 4 submitted
            # | |||||||||||└Action 2 tick 2 used
            # | ||||||||||└Action 2 tick 3 submitted
            # | |||||||||└Action 2 tick 1 used
            # | ||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 2 submitted
            # | |||||||└Action 1 tick 4 used
            # | ||||||└Action 2 tick 1 submitted
            # | |||||└Action 1 tick 3 used
            # | ||||└Action 1 tick 4 submitted
            # | |||└Action 1 tick 2 used
            # | ||└Action 1 tick 3 submitted
            # | |└Action 1 tick 1 used
            # | └Action 1 tick 2 submitted
            # └Obs 1 and 2 created using this state, actions 1 and 2 decided using obs 1 and 2 respectively. Action 1 tick 1 submitted

            # Now for the 1 case. Same issue as above.
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | |||||||||||||||||||||||└Action 3 tick 4 used, etc
            # | ||||||||||||||||||||||└Action 4 tick 1 submitted
            # | |||||||||||||||||||||└Action 3 tick 3 used
            # | ||||||||||||||||||||└Action 3 tick 4 submitted
            # | |||||||||||||||||||└Action 3 tick 2 used
            # | ||||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 3 submitted
            # | |||||||||||||||||└Action 3 tick 1 used
            # | ||||||||||||||||└Action 3 tick 2 submitted
            # | |||||||||||||||└Action 2 tick 4 used
            # | ||||||||||||||└Action 3 tick 1 submitted
            # | |||||||||||||└Action 2 tick 3 used
            # | ||||||||||||└Action 2 tick 4 submitted
            # | |||||||||||└Action 2 tick 2 used
            # | ||||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 3 submitted
            # | |||||||||└Action 2 tick 1 used
            # | ||||||||└Action 2 tick 2 submitted
            # | |||||||└Action 1 tick 4 used
            # | ||||||└Action 2 tick 1 submitted
            # | |||||└Action 1 tick 3 used
            # | ||||└Action 1 tick 4 submitted
            # | |||└Action 1 tick 2 used
            # | ||└Obs 2 created using this state, action 2 decided using obs 2, action 1 tick 3 submitted
            # | |└Action 1 tick 1 used
            # | └Action 1 tick 2 submitted
            # └Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted

            # Now for the -2 case. Here the void state transition actually works in our favor. We do the same thing as in the previous cases using the countdown state
            # before the last countdown state to determine the action, but here the void state transition is equivalent to waiting one tick anyway, so this is tick perfect.
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | |||||||||||||||||||||||└Action 3 tick 4 used, etc
            # | ||||||||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 4 tick 1 submitted
            # | |||||||||||||||||||||└Action 3 tick 3 used
            # | ||||||||||||||||||||└Action 3 tick 4 submitted
            # | |||||||||||||||||||└Action 3 tick 2 used
            # | ||||||||||||||||||└Action 3 tick 3 submitted
            # | |||||||||||||||||└Action 3 tick 1 used
            # | ||||||||||||||||└Action 3 tick 2 submitted
            # | |||||||||||||||└Action 2 tick 4 used
            # | ||||||||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 3 tick 1 submitted
            # | |||||||||||||└Action 2 tick 3 used
            # | ||||||||||||└Action 2 tick 4 submitted
            # | |||||||||||└Action 2 tick 2 used
            # | ||||||||||└Action 2 tick 3 submitted
            # | |||||||||└Action 2 tick 1 used
            # | ||||||||└Action 2 tick 2 submitted
            # | |||||||└Action 1 tick 4 used
            # | ||||||└Obs 2 created using this state, action 2 decided using obs 2, action 2 tick 1 submitted
            # | |||||└Action 1 tick 3 used
            # | ||||└Action 1 tick 4 submitted
            # | |||└Action 1 tick 2 used
            # | ||└Action 1 tick 3 submitted
            # | |└Action 1 tick 1 used
            # | └Action 1 tick 2 submitted
            # └Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted

            # As promised, let's de-clutter by removing all the "action n tick k used" lines.
            # 0 case:
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | | | | | | | | | | | | └Action 4 tick 1 submitted, etc
            # | | | | | | | | | | | └Action 3 tick 4 submitted
            # | | | | | | | | | | └Action 3 tick 3 submitted
            # | | | | | | | | | └Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 2 submitted
            # | | | | | | | | └Action 3 tick 1 submitted
            # | | | | | | | └Action 2 tick 4 submitted
            # | | | | | | └Action 2 tick 3 submitted
            # | | | | | └Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 2 submitted
            # | | | | └Action 2 tick 1 submitted
            # | | | └Action 1 tick 4 submitted
            # | | └Action 1 tick 3 submitted
            # | └Action 1 tick 2 submitted
            # └Obs 1 and 2 created using this state, actions 1 and 2 decided using obs 1 and 2 respectively. Action 1 tick 1 submitted

            # 1 case:
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | | | | | | | | | | | | └Action 4 tick 1 submitted, etc
            # | | | | | | | | | | | └Action 3 tick 4 submitted
            # | | | | | | | | | | └Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 3 submitted
            # | | | | | | | | | └Action 3 tick 2 submitted
            # | | | | | | | | └Action 3 tick 1 submitted
            # | | | | | | | └Action 2 tick 4 submitted
            # | | | | | | └Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 3 submitted
            # | | | | | └Action 2 tick 2 submitted
            # | | | | └Action 2 tick 1 submitted
            # | | | └Action 1 tick 4 submitted
            # | | └Obs 2 created using this state, action 2 decided using obs 2, action 1 tick 3 submitted
            # | └Action 1 tick 2 submitted
            # └Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted

            # -2 case:
            # c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
            # | | | | | | | | | | | | └Obs 4 created using this state, action 4 decided using obs 4, action 4 tick 1 submitted, etc
            # | | | | | | | | | | | └Action 3 tick 4 submitted
            # | | | | | | | | | | └Action 3 tick 3 submitted
            # | | | | | | | | | └Action 3 tick 2 submitted
            # | | | | | | | | └Obs 3 created using this state, action 3 decided using obs 3, action 3 tick 1 submitted
            # | | | | | | | └Action 2 tick 4 submitted
            # | | | | | | └Action 2 tick 3 submitted
            # | | | | | └Action 2 tick 2 submitted
            # | | | | └Obs 2 created using this state, action 2 decided using obs 2, action 2 tick 1 submitted
            # | | | └Action 1 tick 4 submitted
            # | | └Action 1 tick 3 submitted
            # | └Action 1 tick 2 submitted
            # └Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted

            # Let's rewrite this in terms of the number of ticks backwards you have to look based on the length of the previously decided action.
            # Let's say the last action we took defined controller inputs for n ticks, and self._future_tick_action_map defines actions for k more ticks (i.e. k = max(self._future_tick_action_map) - cur_tick).
            # Let's look at when the previous action was created, at tick cur_tick'. We have some k' which is the value of k at the time when the previous action was created. When this happens, the next action
            # will start at cur_tick'+k'+1. After the action gets added, max(self._future_tick_action_map) = cur_tick'+k'+n and so k = cur_tick'+k'+n-cur_tick so k+cur_tick-n+1 = cur_tick'+k'+1.
            # So when k+cur_tick-n+1 = cur_tick (aka n-k=1), we are about to submit the first tick of the last action we generated.
            #
            # In the 0 case, in above_240_fps_mode, we want to generate the new obs when n-k=1 because our most recently decided action is about to start. Otherwise, we want to wait one more tick,
            # so when k+cur_tick-n+2 = cur_tick (aka n-k=2).

            # In the 1 case, in above_240_fps_mode, we want to generate the new obs when n-k=2 because we have then used one tick of the most recently decided action. Otherwise, we want to wait one more tick,
            # so when n-k=3.

            # In the -1 case, which is only achievable in above_240_fps_mode, we want to generate the new obs when k=-1 (i.e. we used the last action in the future tick action map last tick)
            # In the -2 case, in above_240_fps_mode, we want to generate the new obs when k=0, or otherwise when k=-1 (i.e. we used the last action in the future tick action map last tick)

            self._update_gamestate_using_packets(self._unused_packets)
            # We might need to reset if we detect that we have done an incorrect action for any tick we are processing
            if (
                len(self._unused_packets) > 1
                or self._unused_packets[0].match_info.frame_num
                > self._last_packet.match_info.frame_num + 1
            ):
                if not self._check_last_sent_action_correct_for_ticks_since():
                    self.logger.warning(
                        "missed action, recovering using missed action recovery style"
                    )
                    match self.config.missed_action_recovery_style:
                        case MissedActionRecoveryStyle.RESET:
                            self._env_reset_and_update_action_map(
                                packet=self._unused_packets[-1],
                                start_tick=cur_tick,
                                set_state=False,
                            )
                            return
                        case MissedActionRecoveryStyle.IGNORE:
                            # Whatever I guess
                            pass

            last_defined_action_tick = max(self._future_tick_action_map)

            # We need to figure out the next tick on which we need to do an env step
            if self.config.action_step_idx_used_to_build_game_state_for_env_step >= 0:
                # What's the next tick t we would want to create an obs on? It's when self._last_engine_action_length + t - last_defined_action_tick = self.config.action_step_idx_used_to_build_game_state_for_env_step + 1 + (not self.config.above_240_fps_mode)
                next_env_step_tick = (
                    self.config.action_step_idx_used_to_build_game_state_for_env_step
                    + 1
                    + (not self.config.above_240_fps_mode)
                    + last_defined_action_tick
                    - self._last_engine_action_length
                )
            else:
                # last_defined_action_tick - t = 2 - self.config.action_step_idx_used_to_build_game_state_for_env_step - (not self.config.above_240_fps_mode)
                next_env_step_tick = (
                    last_defined_action_tick
                    - 2
                    + self.config.action_step_idx_used_to_build_game_state_for_env_step
                    + (not self.config.above_240_fps_mode)
                )
            if next_env_step_tick < cur_tick:
                # We were supposed to step the env in the past, which might be fine if we still have actions queued up or the action we've been taking is the intended one
                # First, find the game state we should use to step
                step_tick = None
                if next_env_step_tick not in self._hist_game_states_and_packets:
                    self.logger.warning(
                        "missed step tick, recovering using missed step tick recovery style"
                    )
                    # We missed the tick we wanted to use in order to perform the step
                    match self.config.missed_step_tick_recovery_style:
                        case MissedStepTickRecoveryStyle.RESET:
                            self._env_reset_and_update_action_map(
                                packet=self._unused_packets[-1],
                                start_tick=cur_tick,
                                set_state=False,
                            )
                            return
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
                    (obs, *_) = self._env_step(step_gs, step_tick)
                    self._update_future_tick_action_map(
                        step_packet,
                        obs,
                        max(self._future_tick_action_map.keys()) + 1,
                    )
                else:
                    # OK, we still might be fine if the action we have been taking is the intended one, so let's try to step and then see if what we planned aligns with what we actually did
                    (obs, *_) = self._env_step(step_gs, step_tick)
                    self._update_future_tick_action_map(
                        step_packet,
                        obs,
                        max(self._future_tick_action_map.keys()) + 1,
                    )
                    if not self._check_last_sent_action_correct_for_ticks_since():
                        self.logger.warning(
                            "missed action, recovering using missed action recovery style"
                        )
                        match self.config.missed_action_recovery_style:
                            case MissedActionRecoveryStyle.RESET:
                                self._env_reset_and_update_action_map(
                                    packet=self._unused_packets[-1],
                                    start_tick=cur_tick,
                                    set_state=False,
                                )
                                return
                            case MissedActionRecoveryStyle.IGNORE:
                                # Whatever I guess
                                pass
            else:
                if next_env_step_tick == cur_tick:
                    (obs, *_) = self._env_step(self.latest_game_state, cur_tick)
                    self._update_future_tick_action_map(
                        self._unused_packets[-1],
                        obs,
                        max(self._future_tick_action_map.keys()) + 1,
                    )

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
            # If there might be more messages,
            # check for another one with blocking=False
            # if there are no more messages, process the latest packet
            # then wait for the next message with blocking=True
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
                        controller_state = flat.ControllerState()
                        cur_tick = self._last_packet.match_info.frame_num
                        if cur_tick in self._future_tick_action_map:
                            controller_state = self._future_tick_action_map[cur_tick]
                        self._game_interface.send_msg(
                            flat.PlayerInput(self.index, controller_state)
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
        """
        Called when the packet match phase is NOT one of Countdown, Kickoff, or Active. Useful for cosmetic things, like dancing after winning a match or after a goal is scored.
        """
        return flat.ControllerState()

    @abstractmethod
    def get_action(self, obs: ObsType, packet: flat.GamePacket) -> ActionType:
        """
        Called to get action when the actions returned by the last call to the action parser with the last result from this function have all been used up.
        """
        raise NotImplementedError
