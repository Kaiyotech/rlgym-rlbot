from abc import abstractmethod
from typing import Generic

import torch.nn as nn
from rlgym.api import ActionType, ObsType


class Actor(Generic[ObsType, ActionType]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_action(self, obs: ObsType) -> ActionType:
        """
        Function to get an action given an observation
        """
        raise NotImplementedError
