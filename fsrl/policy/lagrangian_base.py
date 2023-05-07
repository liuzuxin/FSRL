from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.utils import MultipleLRSchedulers
from torch import nn
from torch.nn.functional import relu

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.optim_util import LagrangianOptimizer


class LagrangianPolicy(BasePolicy):
    """Implementation of PID Lagrangian-based method.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param List[torch.nn.Module] critics: a list of the critic network. (s -> V(s))
    :param dist_fn: distribution class for computing the action. :type dist_fn:
        Type[torch.distributions.Distribution]
    :param BaseLogger logger: dummy logger for logging events.
    :param bool use_lagrangian: whether to use Lagrangian method. Default to True.
    :param list lagrangian_pid: list of PID constants for Lagrangian multiplier. Default
        to [0.05, 0.0005, 0.1].
    :param float cost_limit: cost limit for the Lagrangian method. Default to np.inf.
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discounting factor for cost and reward, should be in [0, 1].
        Default to 0.99.
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: normalize estimated values to have std close to 1.
        Default to False.
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1]. can be
        either “clip” (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param gym.Space observation_space: environment's observation space. Default to None.
    :param gym.Space action_space: environment's action space. Default to None.
    :param lr_scheduler: learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None.

    .. seealso::

        Please refer to :class:`~fsrl.policy.BasePolicy` for more detailed explanation.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Based policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None
    ) -> None:
        super().__init__(
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.rescaling = rescaling
        self.use_lagrangian = use_lagrangian
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit
        # suppose there are M constraints, then critics_num = M + 1
        if self.use_lagrangian:
            assert len(
                self.cost_limit
            ) == (self.critics_num - 1), "cost_limit must has equal len of critics_num"
            self.lag_optims = [
                LagrangianOptimizer(lagrangian_pid) for _ in range(self.critics_num - 1)
            ]
        else:
            self.lag_optims = []

    def pre_update_fn(self, stats_train: Dict, **kwarg) -> None:
        cost_values = stats_train["cost"]
        self.update_lagrangian(cost_values)

    def update_cost_limit(self, cost_limit: float) -> None:
        """Update the cost limit threshold.

        :param float cost_limit: new cost threshold
        """
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

    def update_lagrangian(self, cost_values: Union[List, float]) -> None:
        """Update the Lagrangian multiplier before updating the policy.

        :param Union[List, float] cost_values: the estimation of cost values that want to
            be controlled under the target thresholds. It could be a list (multiple
            constraints) or a scalar value.
        """
        if np.isscalar(cost_values):
            cost_values = [cost_values]
        for i, lag_optim in enumerate(self.lag_optims):
            lag_optim.step(cost_values[i], self.cost_limit[i])

    def get_extra_state(self):
        """Save the lagrangian optimizer's parameters.

        This function is called when call the policy.state_dict(), see
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state
        """
        if len(self.lag_optims):
            return [optim.state_dict() for optim in self.lag_optims]
        else:
            return None

    def set_extra_state(self, state):
        """Load the lagrangian optimizer's parameters.

        This function is called from load_state_dict() to handle any extra state found
        within the state_dict.
        """
        if "_extra_state" in state:
            lag_optim_cfg = state["_extra_state"]
            if lag_optim_cfg and self.lag_optims:
                for i, state_dict in enumerate(lag_optim_cfg):
                    self.lag_optims[i].load_state_dict(state_dict)

    def safety_loss(self, values: List) -> Tuple[torch.tensor, dict]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lags = [optim.get_lag() for optim in self.lag_optims]
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1. / (np.sum(lags) + 1) if self.rescaling else 1
        assert len(values) == len(lags), "lags and values length must be equal"
        stats = {"loss/rescaling": rescaling}
        loss_safety_total = 0.
        for i, (value, lagrangian) in enumerate(zip(values, lags)):
            loss = torch.mean(value * lagrangian)
            loss_safety_total += loss
            suffix = "" if i == 0 else "_" + str(i)
            stats["loss/lagrangian" + suffix] = lagrangian
            stats["loss/actor_safety" + suffix] = loss.item()
        return loss_safety_total, stats
