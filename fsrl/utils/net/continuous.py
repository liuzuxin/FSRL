# flake8: noqa

from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import Critic
from torch import nn


class DoubleCritic(nn.Module):
    """Double critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net1: a self-defined preprocess_net which output a flattened hidden
        state.
    :param preprocess_net2: a self-defined preprocess_net which output a flattened hidden
        state.
    :param hidden_sizes: a sequence of int for constructing the MLP after preprocess_net.
        Default to empty sequence (where the MLP now contains only a single linear
        layer).
    :param int preprocess_net_output_dim: the output dimension of preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer. Default
        to True.

    For advanced usage (how to customize the network), please refer to tianshou's \
        `build_the_network tutorial <https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network>`_.

    .. seealso::

        Please refer to tianshou's `Net <https://tianshou.readthedocs.io/en/master/api/tianshou.utils.html#tianshou.utils.net.common.Net>`_
        class as an instance of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net1: nn.Module,
        preprocess_net2: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess1 = preprocess_net1
        self.preprocess2 = preprocess_net2
        self.output_dim = 1
        input_dim = getattr(preprocess_net1, "output_dim", preprocess_net_output_dim)
        self.last1 = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )
        self.last2 = deepcopy(self.last1)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> list:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits1, hidden = self.preprocess1(obs)
        logits1 = self.last1(logits1)
        logits2, hidden = self.preprocess2(obs)
        logits2 = self.last2(logits2)
        return [logits1, logits2]

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, list]:
        """Mapping: (s, a) -> logits -> Q(s, a).

        :return: q value, and a list of two q values (used for Bellman backup)"""
        q_list = self(obs, act, info)
        q = torch.min(q_list[0], q_list[1])
        return q, q_list


class SingleCritic(Critic):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value). It differs from
    tianshou's original Critic in that the output will be a list to make the API
    consistent with :class:`~fsrl.utils.net.continuous.DoubleCritic`.

    :param preprocess_net: a self-defined preprocess_net which output a flattened hidden
        state.
    :param hidden_sizes: a sequence of int for constructing the MLP after preprocess_net.
        Default to empty sequence (where the MLP now contains only a single linear
        layer).
    :param int preprocess_net_output_dim: the output dimension of preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer. Default
        to True.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True
    ) -> None:
        super().__init__(
            preprocess_net, hidden_sizes, device, preprocess_net_output_dim,
            linear_layer, flatten_input
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        logits = super().forward(obs, act, info)
        return [logits]

    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, list]:
        """Mapping: (s, a) -> logits -> Q(s, a).

        :return: q value, and a list of two q values (used for Bellman backup)
        """
        q = self(obs, act, info)[0]
        return q, [q]
