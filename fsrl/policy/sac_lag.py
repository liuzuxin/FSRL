from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.exploration import BaseNoise
from torch import nn
from torch.distributions import Independent, Normal

from fsrl.policy.lagrangian_base import LagrangianPolicy
from fsrl.utils import BaseLogger


class SACLagrangian(LagrangianPolicy):
    """Implementation of the Soft Actor-Critic (SAC) with PID Lagrangian.

    More details, please refer to https://arxiv.org/abs/1801.01290 (SAC) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param Optional[torch.optim.Optimizer] actor_optim: the optimizer for the actor
        network.
    :param Optional[torch.optim.Optimizer] critic_optim: the optimizer for the critic
        network(s).
    :param BaseLogger logger: the logger instance for logging training information.
        (default: DummyLogger)
    :param Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] alpha: initial
        temperature for entropy regularization. If a tuple (target_entropy, log_alpha,
        alpha_optim) is provided, then alpha is automatically tuned.(default: 0.005)
    :param float tau: target smoothing coefficient for soft update of target networks.
        (default: 0.05)
    :param Optional[BaseNoise] exploration_noise: the exploration noise. (default: None)
    :param int n_step: number of steps for multi-step learning. (default: 2)
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization.
        (default: True)
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization. (default: [0.05, 0.0005, 0.1])
    :param Union[List, float] cost_limit: the constraint limit(s) for the Lagrangian
        optimization. (default: np.inf)
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discount factor for future rewards. (default: 0.99)
    :param bool reward_normalization: normalize rewards if True. (default: False)
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. (default: True)
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. (default: True)
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). (default: "clip")
    :param Optional[gym.Space] observation_space: the observation space of the
        environment. (default: None)
    :param Optional[gym.Space] action_space: the action space of the environment.
        (default: None)
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. (default: None)


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
        # SAC specific arguments
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.005,
        tau: float = 0.05,
        exploration_noise: Optional[BaseNoise] = None,
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
            rescaling, gamma, 10000, reward_normalization, deterministic_eval,
            action_scaling, action_bound_method, observation_space, action_space,
            lr_scheduler
        )
        self.actor_optim = actor_optim
        self.critics_old = deepcopy(self.critics)
        self.critics_old.eval()
        self.critics_optim = critic_optim
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        self._noise = exploration_noise
        self._n_step = n_step
        self.__eps = np.finfo(np.float32).eps.item()

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
        self.soft_update(self.critics_old, self.critics, self.tau)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> List[torch.Tensor]:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        obs_next_result = self(batch, input='obs_next')
        act = obs_next_result.act
        log_prob = obs_next_result.log_prob
        target_q_list = []
        for i in range(self.critics_num):
            target_q, _ = self.critics_old[i].predict(batch.obs_next, act)
            target_q_list.append(target_q - self._alpha * log_prob)
        return target_q_list

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_returns(
            batch, buffer, indices, self._target_q, self._n_step
        )
        return batch

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian You
        # can check out the original SAC paper (arXiv 1801.01290): Eq 21. in appendix C
        # to get some understanding of this equation.
        squashed_action = torch.tanh(act)
        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
                                        self.__eps).sum(-1, keepdim=True)
        return Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob
        )

    def critics_loss(
        self, batch: Batch, critics: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        loss_critic = 0
        td_average = 0
        stats_critic = {}
        for i in range(self.critics_num):
            target_q = batch.rets[..., i].flatten()
            # double q network
            current_q_list = critics[i](batch.obs, batch.act)
            loss_i = 0
            for j in range(2):
                td = current_q_list[j].flatten() - target_q
                td_average += td
                loss_i += (td.pow(2) * weight).mean()

            loss_critic += loss_i
            stats_critic["loss/q" + str(i)] = loss_i.item()
        optimizer.zero_grad()
        loss_critic.backward()
        optimizer.step()
        td_average /= self.critics_num * 2
        stats_critic["loss/q_total"] = loss_critic.item()
        return td_average, stats_critic

    def policy_loss(self, batch: Batch, **kwarg):
        obs_result = self(batch)
        act = obs_result.act

        # normal loss
        current_q_list = self.critics[0](batch.obs, act)
        current_q = torch.min(current_q_list[0], current_q_list[1]).flatten()
        loss_actor_rew = (self._alpha * obs_result.log_prob.flatten() - current_q).mean()
        # compute safety loss
        values = []
        if self.use_lagrangian:
            for i in range(1, self.critics_num):
                safety_q_list = self.critics[i](batch.obs, act)
                safety_q = torch.min(safety_q_list[0], safety_q_list[1]).flatten()
                values.append(safety_q)

        loss_actor_safety, stats_actor = self.safety_loss(values)

        rescaling = stats_actor["loss/rescaling"]
        loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)

        self.actor_optim.zero_grad()
        loss_actor_total.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
            stats_actor.update(
                {
                    "loss/alpha_loss": alpha_loss.item(),
                    "loss/alpha_value": self._alpha.item()
                }
            )

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
