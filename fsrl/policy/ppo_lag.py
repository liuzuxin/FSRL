from argparse import Action
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn

from fsrl.policy.lagrangian_base import LagrangianPolicy
from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic


class PPOLagrangian(LagrangianPolicy):
    """Implementation of the Proximal Policy Optimization (PPO) with PID Lagrangian.

    More details, please refer to https://arxiv.org/abs/1707.06347 (PPO) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for the actor and critic networks.
    :param Type[torch.distributions.Distribution] dist_fn: the distribution class for
        action sampling.
    :param BaseLogger logger: the logger instance for logging training information.
        Default is BaseLogger().
    :param float target_kl: the target KL divergence for the PPO update. Default is 0.02.
    :param float vf_coef: the value function coefficient for the loss function. Default
        is 0.25.
    :param Optional[float] max_grad_norm: the maximum gradient norm for gradient clipping
        (None for no clipping). Default is None.
    :param float gae_lambda: the Generalized Advantage Estimation (GAE) parameter.
        Default is 0.95.
    :param float eps_clip: the PPO clipping parameter for the policy update. Default is
        0.2.
    :param Optional[float] dual_clip: the PPO dual clipping parameter (None for no dual
        clipping). Default is None.
    :param bool value_clip: whether to clip the value function update. Default is False.
    :param bool advantage_normalization: whether to normalize the advantages. Default is
        True.
    :param bool recompute_advantage: whether to recompute the advantages during the
        optimization process. Default is False.
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization.
        Default is True.
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization. Default is [0.05, 0.0005, 0.1].
    :param Union[List, float] cost_limit: the constraint limit(s) for the Lagrangian
        optimization. Default is np.inf.
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discount factor for future rewards. Default is 0.99.
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: whether to normalize the rewards. Default is False.
    :param bool deterministic_eval: whether to use deterministic actions during
        evaluation. Default is True.
    :param bool action_scaling: whether to scale actions based on the action space.
        Default is True.
    :param str action_bound_method: the method used to handle out-of-bound actions.
        Default is "clip".
    :param Optional[gym.Space] observation_space: the observation space of the
        environment. Default is None.
    :param Optional[gym.Space] action_space: the action space of the environment. Default
        is None.
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: the learning rate
        scheduler. Default is None.

    .. seealso::

       Please refer to :class:`~fsrl.policy.BasePolicy` and
       :class:`~fsrl.policy.LagrangianPolicy` for more detailed \
           hyperparameter explanations and usage.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # PPO specific argumentsß
        target_kl: float = 0.02,
        vf_coef: float = 0.25,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        cost_limit: Union[List, float] = np.inf,
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        super().__init__(
            actor, critics, dist_fn, logger, use_lagrangian, lagrangian_pid, cost_limit,
            rescaling, gamma, max_batchsize, reward_normalization, deterministic_eval,
            action_scaling, action_bound_method, observation_space, action_space,
            lr_scheduler
        )
        self.optim = optim
        self._lambda = gae_lambda
        self._weight_vf = vf_coef
        self._grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        # batch get 3 new keys: values, rets, advs
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def critics_loss(self, minibatch):
        critic_losses = 0
        stats = {}
        for i, critic in enumerate(self.critics):
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            if self._value_clip:
                value_target = minibatch.values[..., i]
                v_clip = value_target + \
                    (value - value_target).clamp(-self._eps_clip, self._eps_clip)
                vf1 = (ret - value).pow(2)
                vf2 = (ret - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (ret - value).pow(2).mean()
            critic_losses += vf_loss

            stats["loss/vf" + str(i)] = vf_loss.item()
        stats["loss/vf_total"] = critic_losses.item()
        return critic_losses, stats

    def policy_loss(self, batch: Batch, dist: Type[torch.distributions.Distribution]):

        log_p = dist.log_prob(batch.act)
        ratio = (log_p - batch.logp_old).exp().float()
        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = batch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                batch.advs[..., i] = (adv - mean) / std  # per-batch norm

        # compute normal ppo loss
        rew_adv = batch.advs[..., 0]
        surr1 = ratio * rew_adv
        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * rew_adv
        if self._dual_clip:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self._dual_clip * rew_adv)
            loss_actor_rew = -torch.where(rew_adv < 0, clip2, clip1).mean()
        else:
            loss_actor_rew = -torch.min(surr1, surr2).mean()

        # compute safety loss
        values = [ratio * batch.advs[..., i]
                  for i in range(1, self.critics_num)] if self.use_lagrangian else []
        loss_actor_safety, stats_actor = self.safety_loss(values)

        rescaling = stats_actor["loss/rescaling"]
        loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)

        # compute approx KL for early stop
        approx_kl = (batch.logp_old - log_p).mean().item()
        stats_actor.update(
            {
                "loss/actor_rew": loss_actor_rew.item(),
                "loss/actor_total": loss_actor_total.item(),
                "loss/kl": approx_kl
            }
        )
        return loss_actor_total, stats_actor

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self.compute_gae_returns(
                    batch, self._buffer, self._indices, self._lambda
                )
            iter_counts, approx_kl = 0, 0.0
            for minibatch in batch.split(batch_size, merge_last=True):
                # obtain the action distribution
                dist = self.forward(minibatch).dist
                # calculate policy loss
                loss_actor, stats_actor = self.policy_loss(minibatch, dist)
                approx_kl += stats_actor["loss/kl"]
                iter_counts += 1
                # calculate loss for critic
                loss_vf, stats_critic = self.critics_loss(minibatch)

                loss = loss_actor + self._weight_vf * loss_vf

                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                self.gradient_steps += 1

                ent = dist.entropy().mean()
                self.logger.store(**stats_actor)
                self.logger.store(**stats_critic)
                self.logger.store(total=loss.item(), entropy=ent.item(), tab="loss")

            # trick in
            # https://github.com/liuzuxin/robust-safe-rl/blob/main/rsrl/policy/robust_ppo.py # noqa
            approx_kl /= iter_counts + 1e-7
            if approx_kl > 1.5 * self._target_kl:
                # early stop
                self.logger.print("Early stop at step %d due to reaching max kl." % step)
                break

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")
