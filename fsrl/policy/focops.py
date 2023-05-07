import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn
from torch.distributions import kl_divergence

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic


class FOCOPS(BasePolicy):
    """Implementation of the First Order Constrained Optimization in Policy Space.
    
    More details, please refer to https://arxiv.org/pdf/2002.06506.pdf

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param torch.optim.Optimizer actor_optim: the optimizer for the actor network.
    :param torch.optim.Optimizer critic_optim: the optimizer for the critic network(s).
    :param Type[torch.distributions.Distribution] dist_fn: the probability distribution
        function for sampling actions.
    :param BaseLogger logger: the logger instance for logging training information.
    :param float cost_limit: the constraint limit for the optimization. Default value is
        10.
    :param Union[float, Tuple[float, float, torch.Tensor]] nu: cost coefficient. Default
        value is 0.01.
    :param float l2_reg: L2 regularization rate. Default value is 1e-3.
    :param float delta: early stop KL bound. Default value is 0.02.
    :param float eta: KL bound for indicator function. Default value is 0.02.
    :param float tem_lambda: inverse temperature lambda. Default value is 0.95.
    :param float gae_lambda: GAE (Generalized Advantage Estimation) lambda for advantage
        computation. Default value is 0.95.
    :param Optional[float] max_grad_norm: maximum gradient norm for gradient clipping, if
        specified. Default value is 0.5.
    :param bool advantage_normalization: normalize advantage if True. Default value is
        True.
    :param bool recompute_advantage: recompute advantage using the updated value
        function. Default value is False.
    :param float gamma: the discount factor for future rewards. Default value is 0.99.
    :param int max_batchsize: maximum batch size for the optimization. Default value is
        99999.
    :param bool reward_normalization: normalize the rewards if True. Default value is
        False.
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. Default value is True.
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. Default value is True.
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). Default value is "clip".
    :param Optional[gym.Space] observation_space: the observation space of the
        environment. Default value is None.
    :param Optional[gym.Space] action_space: the action space of the environment. Default
        value is None.
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. Default value is None.

    .. seealso::

        Please refer to :class:`~fsrl.policy.BasePolicy` for more detailed hyperparameter
        explanations and usage.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        nu: Union[float, Tuple[float, float, torch.Tensor]] = 0.01,
        l2_reg: float = 1e-3,
        delta: float = 0.02,
        eta: float = 0.02,
        tem_lambda: float = 0.95,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = 0.5,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
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
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.actor_optim = actor_optim
        self.critics_optim = critic_optim
        self.cost_limit = cost_limit
        self._gae_lambda = gae_lambda
        self._tem_lambda = tem_lambda
        self._grad_norm = max_grad_norm
        self._is_auto_nu = False
        if isinstance(nu, tuple):
            self._is_auto_nu = True
            self._nu_max, self._nu_lr, self._nu = nu
        else:
            self._nu = nu
        self._l2_reg = l2_reg
        self._delta = delta
        self._eta = eta
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic

    def pre_update_fn(self, stats_train: Dict, **kwarg) -> Any:
        self._ave_cost_return = stats_train["cost"]

    def update_cost_limit(self, cost_limit: float) -> None:
        """Update the cost limit threshold.

        :param float cost_limit: new cost threshold
        """
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        # batch get 3 new keys: values, rets, advs
        batch = self.compute_gae_returns(batch, buffer, indices, self._gae_lambda)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob, old_mean, old_std = [], [], []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                res = self.forward(minibatch)
                old_log_prob.append(res.dist.log_prob(minibatch.act))
                old_mean.append(res.logits[0, ...])
                old_std.append(res.logits[1, ...])
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.mean_old = torch.cat(old_mean, dim=0)
        batch.std_old = torch.cat(old_std, dim=0)
        return batch

    def nu_loss(self, batch: Batch):
        loss_nu = self.cost_limit - self._ave_cost_return
        self._nu += -self._nu_lr * loss_nu
        self._nu = torch.clamp(self._nu, 0, self._nu_max)
        stats_nu = {"loss/nu_loss": loss_nu, "loss/nu_value": self._nu.detach().item()}
        return loss_nu, stats_nu

    def critics_loss(self, minibatch: Batch):
        critic_losses = 0
        stats = {}
        for i, critic in enumerate(self.critics):
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            vf_loss = (ret - value).pow(2).mean()
            for param in critic.parameters():
                vf_loss += param.pow(2).sum() * self._l2_reg
            critic_losses += vf_loss
            stats["loss/vf" + str(i)] = vf_loss.item()

        self.critics_optim.zero_grad()
        critic_losses.backward()
        self.critics_optim.step()
        stats["loss/vf_total"] = critic_losses.item()
        return critic_losses, stats

    def policy_loss(self, minibatch: Batch):
        # obtain the action distribution
        dist = self.forward(minibatch).dist
        ent = dist.entropy().mean()

        log_p = dist.log_prob(minibatch.act)
        ratio = (log_p - minibatch.logp_old).exp()
        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))
        kl_new_old = kl_divergence(dist, dist_old)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = minibatch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                minibatch.advs[..., i] = (adv - mean) / std  # per-batch norm

        rew_adv = minibatch.advs[..., 0]
        cost_adv = minibatch.advs[..., 1]
        loss_actor = (
            (
                kl_new_old - 1 / self._tem_lambda * ratio *
                (rew_adv - self._nu * cost_adv)
            ) * (kl_new_old.detach() <= self._eta)
        ).mean()

        self.actor_optim.zero_grad()
        loss_actor.backward()
        if self._grad_norm:  # clip large gradient
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self._grad_norm)
        self.actor_optim.step()
        stats_actor = {
            "loss/actor_loss": loss_actor.item(),
            "loss/kl": kl_new_old.mean().item(),
            "loss/entropy": ent.item()
        }
        return loss_actor, stats_actor

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:
        # update nu
        loss_nu, stats_nu = self.nu_loss(batch)

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self.compute_gae_returns(
                    batch, self._buffer, self._indices, self._gae_lambda
                )
            iter_counts, approx_kl = 0, 0.0
            for minibatch in batch.split(batch_size, merge_last=True):
                # update critic
                loss_vf, stats_critic = self.critics_loss(minibatch)

                # update actor
                loss_actor, stats_actor = self.policy_loss(minibatch)

                approx_kl += stats_actor["loss/kl"]
                iter_counts += 1
                self.gradient_steps += 1

                self.logger.store(**stats_nu)
                self.logger.store(**stats_actor)
                self.logger.store(**stats_critic)

            # trick in
            # https://github.com/liuzuxin/robust-safe-rl/blob/main/rsrl/policy/robust_ppo.py # noqa: E501
            approx_kl /= iter_counts + 1e-7
            if approx_kl > self._delta:
                # early stop
                self.logger.print("Early stop at step %d due to reaching max kl." % step)
                break

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")
