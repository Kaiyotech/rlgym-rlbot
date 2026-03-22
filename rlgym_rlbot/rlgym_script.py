from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass
from traceback import print_exc
from typing import Any, Callable, Dict, Generic, List, Optional

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


@dataclass
class RLGymBotConfig:
    v1_style_tick_skip = False
    standard_map = True
    sim_extra_info = False


@dataclass
class RLGymEnvPersistentState(Generic[ActionType, ObsType, RewardType]):
    game_interface: SocketRelay
    config: RLGymBotConfig
    game_state: GameState
    spawn_id: int
    get_action_fn: Callable[[ObsType, flat.GamePacket], ActionType]
    obs_builder: ObsBuilder[AgentID, ObsType, GameState, Any]
    action_parser: ActionParser[AgentID, ActionType, np.ndarray, GameState, Any]
    reward_function: Optional[RewardFunction[AgentID, GameState, RewardType]] = None
    termination_condition: Optional[DoneCondition[AgentID, GameState]] = None
    truncation_condition: Optional[DoneCondition[AgentID, GameState]] = None
    state_mutator: Optional[StateMutator[GameState]] = None
    shared_info_provider: Optional[SharedInfoProvider[AgentID, GameState]] = None
    sim_extra_info: Optional[SimExtraInfo] = None
    shared_info: Dict[str, Any] = {}
    future_tick_action_map: Dict[int, flat.ControllerState] = {}


class RLGymEnvState(Generic[ActionType, ObsType, RewardType]):
    def __init__(
        self,
        persistent_state: RLGymEnvPersistentState[ActionType, ObsType, RewardType],
    ):
        self.persistent_state = persistent_state
        self.game_interface = persistent_state.game_interface
        self.config = persistent_state.config
        self.game_state = persistent_state.game_state
        self.spawn_id = persistent_state.spawn_id
        self.get_action_fn = persistent_state.get_action_fn
        self.obs_builder = persistent_state.obs_builder
        self.action_parser = persistent_state.action_parser
        self.reward_function = persistent_state.reward_function
        self.termination_condition = persistent_state.termination_condition
        self.truncation_condition = persistent_state.truncation_condition
        self.state_mutator = persistent_state.state_mutator
        self.shared_info_provider = persistent_state.shared_info_provider
        self.sim_extra_info = persistent_state.sim_extra_info

    def handle_packet(
        self, packet: flat.GamePacket, index: int
    ) -> RLGymEnvState[ActionType, ObsType, RewardType]:
        pass

    def update_game_state(self, packet: flat.GamePacket):
        extra_info = None
        if self.config.sim_extra_info:
            extra_info = self.sim_extra_info.get_extra_info(packet)
        self.game_state.update(packet, extra_info)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__


class ResetPhase1State(RLGymEnvState[ActionType, ObsType, RewardType]):
    def handle_packet(self, packet, index):
        self.update_game_state(packet)
        if self.shared_info_provider is not None:
            self.persistent_state.shared_info = self.shared_info_provider.create(
                self.persistent_state.shared_info
            )
        if self.state_mutator is not None:
            # Use state mutator to define new state and progress to reset phase 2
            desired_state = create_base_state()
            self.state_mutator.apply(desired_state, self.persistent_state.shared_info)
            self.game_interface.send_game_state(
                gamestate_rlgym_to_rlbot(desired_state, packet)
            )
            return ResetPhase2State(self.persistent_state)
        # Since there was no state mutator, we will just continue with whatever RLBot has given us
        # TODO: it's ugly to duplicate all of this from ResetPhase2State just to avoid the second update_game_state call, but I don't want to think about refactoring yet
        agents = list(self.game_state.cars.keys())
        cur_tick = packet.match_info.frame_num
        if self.shared_info_provider is not None:
            self.persistent_state.shared_info = self.shared_info_provider.set_state(
                agents, self.game_state, self.persistent_state.shared_info
            )
        self.obs_builder.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        self.action_parser.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        if self.termination_condition is not None:
            self.termination_condition.reset(
                agents, self.game_state, self.persistent_state.shared_info
            )
        if self.truncation_condition is not None:
            self.truncation_condition.reset(
                agents, self.game_state, self.persistent_state.shared_info
            )
        self.reward_function.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        obs = self.obs_builder.build_obs(
            agents, self.game_state, self.persistent_state.shared_info
        )
        action = self.get_action_fn(obs[self.spawn_id], packet)
        engine_action = self.action_parser.parse_actions(
            {self.spawn_id: action}, self.game_state, self.persistent_state.shared_info
        )[self.spawn_id]
        steps = engine_action.shape[0]
        for idx in range(steps):
            self.persistent_state.future_tick_action_map[cur_tick + idx] = (
                flat.ControllerState(*engine_action[idx], False)
            )
        self.game_interface.send_player_input(
            flat.PlayerInput(
                index, self.persistent_state.future_tick_action_map[cur_tick]
            )
        )
        return PotentialStepState(self.persistent_state)


class ResetPhase2State(RLGymEnvState[ActionType, ObsType, RewardType]):
    def handle_packet(self, packet, index):
        self.update_game_state(packet)
        # TODO: maybe agents should only be this agent
        agents = list(self.game_state.cars.keys())
        cur_tick = packet.match_info.frame_num
        if self.shared_info_provider is not None:
            self.persistent_state.shared_info = self.shared_info_provider.set_state(
                agents, self.game_state, self.persistent_state.shared_info
            )
        self.obs_builder.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        self.action_parser.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        if self.termination_condition is not None:
            self.termination_condition.reset(
                agents, self.game_state, self.persistent_state.shared_info
            )
        if self.truncation_condition is not None:
            self.truncation_condition.reset(
                agents, self.game_state, self.persistent_state.shared_info
            )
        self.reward_function.reset(
            agents, self.game_state, self.persistent_state.shared_info
        )
        obs = self.obs_builder.build_obs(
            agents, self.game_state, self.persistent_state.shared_info
        )
        action = self.get_action_fn(obs[self.spawn_id], packet)
        engine_action = self.action_parser.parse_actions(
            {self.spawn_id: action}, self.game_state, self.persistent_state.shared_info
        )[self.spawn_id]
        steps = engine_action.shape[0]
        for idx in range(steps):
            self.persistent_state.future_tick_action_map[cur_tick + idx] = (
                flat.ControllerState(*engine_action[idx], False)
            )
        self.game_interface.send_player_input(
            flat.PlayerInput(
                index, self.persistent_state.future_tick_action_map[cur_tick]
            )
        )
        return PotentialStepState(self.persistent_state)


# TODO: all of these states don't account for the fact that we might not be processing the latest packet
class PotentialStepState(RLGymEnvState[ActionType, ObsType, RewardType]):
    def handle_packet(self, packet: flat.GamePacket, index: int):
        self.update_game_state(packet)
        # TODO: implement based on whether future tick action map has value for current tick
        cur_tick = packet.match_info.frame_num
        if cur_tick in self.persistent_state.future_tick_action_map:
            self.game_interface.send_player_input(
                flat.PlayerInput(
                    index, self.persistent_state.future_tick_action_map[cur_tick]
                )
            )
            return self
        # TODO: maybe agents should only be this agent
        agents = list(self.game_state.cars.keys())

        if self.shared_info_provider is not None:
            self.persistent_state.shared_info = self.shared_info_provider.step(
                agents, self.game_state, self.persistent_state.shared_info
            )
        obs = self.obs_builder.build_obs(
            agents, self.game_state, self.persistent_state.shared_info
        )
        is_terminated = {self.spawn_id: False}
        is_truncated = {self.spawn_id: False}
        if self.termination_condition is not None:
            is_terminated = self.termination_condition.is_done(
                agents, self.game_state, self.persistent_state.shared_info
            )
        if self.truncation_condition is not None:
            is_truncated = self.truncation_condition.is_done(
                agents, self.game_state, self.persistent_state.shared_info
            )
        # is_terminated = self.termination_cond.is_done(agents, new_state, self.persistent_state.shared_info) \
        #     if self.termination_cond is not None else {agent: False for agent in agents}
        # is_truncated = self.truncation_cond.is_done(agents, new_state, self.persistent_state.shared_info) \
        #     if self.truncation_cond is not None else {agent: False for agent in agents}
        # rewards = self.reward_fn.get_rewards(agents, new_state, is_terminated, is_truncated, self.persistent_state.shared_info)
        # TODO: finish
        pass


class RLGymBot(Generic[ActionType, ObsType, RewardType]):
    """
    A convenience base class for bots developed using RLGym.
    The base class handles the setup and communication with the rlbot server, along with management of RLGym config objects.
    Subclass from this to override the following methods:
    - initialize
    - retire
    - get_action
    - handle_match_comm

    """

    logger = DEFAULT_LOGGER

    team: int = -1
    index: int = -1
    name: str = ""
    spawn_id: int = 0

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

    _latest_packet: Optional[flat.GamePacket] = None
    _latest_prediction = flat.BallPrediction()

    def __init__(
        self,
        tick_skip: int,
        obs_builder: ObsBuilder[AgentID, ObsType, GameState, Any],
        action_parser: ActionParser[AgentID, ActionType, np.ndarray, GameState, Any],
        reward_function: Optional[
            RewardFunction[AgentID, GameState, RewardType]
        ] = None,
        termination_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        truncation_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        state_mutator: Optional[StateMutator[GameState]] = None,
        shared_info_provider: Optional[SharedInfoProvider[AgentID, GameState]] = None,
        # TODO: rename to whatever this is in 0.7
        default_agent_id: Optional[str] = None,
        v1_style_tick_skip=False,
        standard_map=True,
        sim_extra_info=False,
    ):
        self.tick_skip = tick_skip
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.state_mutator = state_mutator
        self.shared_info_provider = shared_info_provider
        self.v1_style_tick_skip = v1_style_tick_skip
        self.standard_map = standard_map
        self.sim_extra_info = sim_extra_info
        self.received_first_packet = False
        self.shared_info = (
            shared_info_provider.create({}) if shared_info_provider is not None else {}
        )
        self.future_tick_action_map = {}

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
            if player.spawn_id == self.spawn_id:
                self.name = player.name
                self.logger = get_logger(self.name)
                break

        self.game_state = GameState.create_compat_game_state(
            self.field_info, self.match_config, self.tick_skip, self.standard_map
        )
        if self.sim_extra_info:
            self.sim_extra_info = SimExtraInfo(
                self.field_info, self.match_config, self.tick_skip
            )
        try:
            self.initialize()
        except Exception as e:
            self.logger.critical(
                "Bot %s failed to initialize due the following error: %s", self.name, e
            )
            print_exc()
            exit()

        self._initialized_bot = True
        self._game_interface.send_init_complete()

    def _handle_match_config(self, match_config: flat.MatchConfiguration):
        self.match_config = match_config
        self.agent_id_map = {
            p.spawn_id: p.agent_id for p in self.match_config.player_configurations
        }
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
        self.spawn_id = controllable.spawn_id
        self.index = controllable.index
        self._has_player_mapping = True

        self._try_initialize()

    def _handle_ball_prediction(self, ball_prediction: flat.BallPrediction):
        self._latest_prediction = ball_prediction

    def _update_future_tick_action_map(self, obs: Dict[int, ObsType], cur_tick: int):
        self.future_tick_action_map.clear()
        action = self.get_action(obs[self.spawn_id])
        engine_action = self.action_parser.parse_actions(
            {self.spawn_id: action}, self.game_state, self.shared_info
        )[self.spawn_id]
        steps = engine_action.shape[0]
        for idx in range(steps):
            self.future_tick_action_map[cur_tick + idx] = flat.ControllerState(
                *engine_action[idx], False
            )

    def _handle_packet(self, packet: flat.GamePacket):
        self._latest_packet = packet
        try:
            # TODO: I think I should only be doing this during countdown, kickoff, and active. Fork over control back to the user if in other phase? What about the fact that this might not be the latest packet?
            extra_info = None
            if self.sim_extra_info:
                extra_info = self.sim_extra_info.get_extra_info(packet)
            self.game_state.update(packet, extra_info)
            if len(packet.players) <= self.index:
                return
            agents = list(self.game_state.cars.keys())
            obs = None
            cur_tick = packet.match_info.frame_num

            # TODO: maybe agents should only be this agent
            # If this is the first packet, we need to emulate RLGym's reset() functionality
            if not self.received_first_packet:
                self.received_first_packet = True
                if self.shared_info_provider is not None:
                    self.shared_info = self.shared_info_provider.create(
                        self.shared_info
                    )
                # TODO: GameState -> GamePacket + state setting for state mutator
                desired_state = create_base_state()
                self.state_mutator.apply(desired_state, self.shared_info)
                # state = self.transition_engine.set_state(desired_state, self.shared_info)
                if self.shared_info_provider is not None:
                    self.shared_info = self.shared_info_provider.set_state(
                        agents,
                        self.game_state,
                        self.shared_info,
                    )
                self.obs_builder.reset(agents, self.game_state, self.shared_info)
                self.action_parser.reset(agents, self.game_state, self.shared_info)
                if self.termination_condition is not None:
                    self.termination_condition.reset(
                        agents, self.game_state, self.shared_info
                    )
                if self.truncation_condition is not None:
                    self.truncation_condition.reset(
                        agents, self.game_state, self.shared_info
                    )
                if self.reward_function is not None:
                    self.reward_function.reset(
                        agents, self.game_state, self.shared_info
                    )
                obs = self.obs_builder.build_obs(
                    agents, self.game_state, self.shared_info
                )
                self._update_future_tick_action_map(obs, cur_tick)

            # TODO: this should really
            # Simulate RLGym's step() functionality
            if cur_tick not in self.future_tick_action_map:
                if self.shared_info_provider is not None:
                    self.shared_info = self.shared_info_provider.step(
                        agents, self.game_state, self.shared_info
                    )
                obs = self.obs_builder.build_obs(
                    agents, self.game_state, self.shared_info
                )
                is_terminated = (
                    self.termination_condition.is_done(
                        agents, self.game_state, self.shared_info
                    )
                    if self.termination_condition is not None
                    else {agent: False for agent in agents}
                )
                is_truncated = (
                    self.truncation_condition.is_done(
                        agents, self.game_state, self.shared_info
                    )
                    if self.truncation_condition is not None
                    else {agent: False for agent in agents}
                )
                if is_terminated or is_truncated:
                    # TODO: reset, including state mutator
                    pass
                if self.reward_function is not None:
                    rewards = self.reward_function.get_rewards(
                        agents,
                        self.game_state,
                        is_terminated,
                        is_truncated,
                        self.shared_info,
                    )
                    self.handle_reward(rewards[self.spawn_id])
                self._update_future_tick_action_map(obs, cur_tick)

            player_input = flat.PlayerInput(
                self.index, self.future_tick_action_map[cur_tick]
            )
            self._game_interface.send_player_input(player_input)
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
            match self._game_interface.handle_incoming_messages(blocking=block_next):
                case MsgHandlingResult.TERMINATED:
                    running = False
                case MsgHandlingResult.NO_INCOMING_MSGS:
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
        """Called if reward_function is not None each time a packet is received"""

    @abstractmethod
    def get_action(self, obs: ObsType, packet: flat.GamePacket) -> ActionType:
        """
        Called to get action when the actions returned by the last call to the action parser with the last result from this function have all been used up.
        """
        raise NotImplementedError
