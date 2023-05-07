from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.exploration import BaseNoise, GaussianNoise
from torch import nn

from fsrl.policy.lagrangian_base import LagrangianPolicy
from fsrl.utils import BaseLogger


class DDPGLagrangian(LagrangianPolicy):
    """The Deep Deterministic Policy Gradient (DDPG) with PID Lagrangian.

    More details, please refer to https://arxiv.org/abs/1509.02971 (DDPG) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param Optional[torch.optim.Optimizer] actor_optim: the optimizer for the actor
        network. Default is None.
    :param Optional[torch.optim.Optimizer] critic_optim: the optimizer for the critic
        network(s). Default is None.
    :param BaseLogger logger: the logger instance for logging training information.
        Default is DummyLogger.
    :param float tau: the soft update coefficient for updating target networks. Default
        is 0.05.
    :param Optional[BaseNoise] exploration_noise: the noise instance for exploration.
        Default is GaussianNoise(sigma=0.1).
    :param int n_step: the number of steps for multi-step bootstrap targets. Default is
        2.
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization.
        Default is True.
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization. Default is [0.05, 0.0005, 0.1].
    :param Union[List, float] cost_limit: the constraint limit(s) for the Lagrangian
        optimization. Default is np.inf.
    :param bool rescaling: whether to rescale the Lagrangian multiplier. Default is True.
    :param float gamma: the discount factor for future rewards. Default is 0.99.
    :param bool reward_normalization: normalize rewards if True. Default is False.
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. Default is True.
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. Default is True.
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). Default is "clip".
    :param Optional[gym.Space] observation_space: the observation space of the
        environment. Default is None.
    :param Optional[gym.Space] action_space: the action space of the environment. Default
        is None.
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. Default is None.

    .. seealso::

        Please refer to :class:`~fsrl.policy.BasePolicy` and
        :class:`~fsrl.policy.LagrangianPolicy` for more detailed \
           hyperparameter explanations and usage.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        actor_optim: Optional[torch.optim.Optimizer],
        critic_optim: Optional[torch.optim.Optimizer],
        logger: BaseLogger = BaseLogger(),
        # DDPG specific arguments
        tau: float = 0.05,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        n_step: int = 2,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        super().__init__(
            actor, critics, None, logger, use_lagrangian, lagrangian_pid, cost_limit,
            rescaling, gamma, 99999, reward_normalization, deterministic_eval,
            action_scaling, action_bound_method, observation_space, action_space,
            lr_scheduler
        )

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critics_old = deepcopy(self.critics)
        self.critics_old.eval()
        self.critics_optim = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        self._noise = exploration_noise
        self._n_step = n_step

    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    def train(self, mode: bool = True):
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.actor.train(mode)
        self.critics.train(mode)
        return self

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critics_old, self.critics, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> List[torch.Tensor]:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        action_next = self(batch, model='actor_old', input='obs_next').act
        target_q = []
        for i in range(self.critics_num):
            target_q.append(self.critics_old[i](batch.obs_next, action_next))
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_returns(
            batch, buffer, indices, self._target_q, self._n_step
        )
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for more detailed
            explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        actions, hidden = model(obs, state=state, info=batch.info)
        return Batch(act=actions, state=hidden)

    def critics_loss(
        self, batch: Batch, critics: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        loss_critic = 0
        td_average = 0
        stats_critic = {}
        for i in range(self.critics_num):
            current_q = critics[i](batch.obs, batch.act).flatten()
            target_q = batch.rets[..., i].flatten()
            td = current_q - target_q
            td_average += td
            loss_i = (td.pow(2) * weight).mean()
            loss_critic += loss_i
            stats_critic["loss/q" + str(i)] = loss_i.item()
        optimizer.zero_grad()
        loss_critic.backward()
        optimizer.step()
        td_average /= self.critics_num
        stats_critic["loss/q_total"] = loss_critic.item()
        return td_average, stats_critic

    def policy_loss(self, batch: Batch, **kwarg):
        action = self(batch, model="actor", input="obs").act
        # normal loss
        loss_actor_rew = -self.critics[0](batch.obs, action).mean()
        # compute safety loss
        values = [
            self.critics[i](batch.obs, action).mean()
            for i in range(1, self.critics_num)
        ] if self.use_lagrangian else []
        loss_actor_safety, stats_actor = self.safety_loss(values)

        rescaling = stats_actor["loss/rescaling"]
        loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)

        self.actor_optim.zero_grad()
        loss_actor_total.backward()
        self.actor_optim.step()

        stats_actor.update(
            {
                "loss/actor_rew": loss_actor_rew.item(),
                "loss/actor_total": loss_actor_total.item()
            }
        )
        return loss_actor_total, stats_actor

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic
        td, stats_critic = self.critics_loss(batch, self.critics, self.critics_optim)
        batch.weight = td  # prio-buffer
        # actor
        loss_actor, stats_actor = self.policy_loss(batch)
        self.sync_weight()
        self.logger.store(**stats_actor)
        self.logger.store(**stats_critic)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        return act
