from typing import List, Union

import torch.nn as nn


class ActorCritic(nn.Module):
    """An actor-critic network for parsing parameters.

    :param nn.Module actor: the actor network.
    :param nn.Module critic: the critic network.
    """

    def __init__(self, actor: nn.Module, critics: Union[List, nn.Module]):
        super().__init__()
        self.actor = actor
        if isinstance(critics, List):
            critics = nn.ModuleList(critics)
        self.critics = critics
