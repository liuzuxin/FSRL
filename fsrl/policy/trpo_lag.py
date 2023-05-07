from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn
from torch.distributions import kl_divergence

from fsrl.policy.lagrangian_base import LagrangianPolicy
from fsrl.utils import BaseLogger


class TRPOLagrangian(LagrangianPolicy):
    """Implementation of the Trust Region Policy Optimization (TRPO) with PID Lagrangian.

    More details, please refer to https://arxiv.org/abs/1502.05477 (TRPO) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param Type[torch.distributions.Distribution] dist_fn: the distribution function for
        the policy.
    :param BaseLogger logger: the logger instance for logging training information.
    :param float target_kl: the target KL divergence for the line search (default:
        0.001).
    :param float backtrack_coeff: the coefficient for backtracking during the line search
        (default: 0.8).
    :param int max_backtracks: the maximum number of backtracks allowed during the line
        search (default: 10).
    :param int optim_critic_iters: the number of optimization iterations for the critic
        network (default: 5).
    :param float gae_lambda: the GAE lambda value (default: 0.95).
    :param bool advantage_normalization: whether to normalize advantage (default: True).
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization
        (default: True).
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization (default: [0.05, 0.0005, 0.1]).
    :param Union[List, float] cost_limit: the constraint limit(s) for the Lagrangian
        optimization (default: np.inf).
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discount factor for future rewards (default: 0.99).
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: whether to normalize rewards (default: False).
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation (default: True).
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds (default: True).
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods) (default: "clip").
    :param Optional[gym.Space] observation_space: the observation space of the
        environment (default: None).
    :param Optional[gym.Space] action_space: the action space of the environment
        (default: None).
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer (default: None).

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
        # TRPO specific argumentsß
        target_kl: float = 0.001,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        optim_critic_iters: int = 5,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
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
        self._norm_adv = advantage_normalization
        self._max_backtracks = max_backtracks
        self._delta = target_kl
        self._backtrack_coeff = backtrack_coeff
        self._optim_critic_iters = optim_critic_iters
        # adjusts Hessian-vector product calculation for numerical stability
        self._damping = 0.1

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        batch.act = to_torch_as(batch.act, batch.values[..., 0])
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = batch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                batch.advs[..., i] = (adv - mean) / std  # per-batch norm
        return batch

    def critics_loss(self, minibatch):
        critic_losses = 0
        stats = {}
        for i, critic in enumerate(self.critics):
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            vf_loss = (ret - value).pow(2).mean()
            critic_losses += vf_loss
            stats["loss/vf" + str(i)] = vf_loss.item()
        stats["loss/vf_total"] = critic_losses.item()
        return critic_losses, stats

    def policy_loss(self, batch: Batch, dist: Type[torch.distributions.Distribution]):

        log_p = dist.log_prob(batch.act)
        ratio = (log_p - batch.logp_old).exp().float()
        ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)

        rew_adv = batch.advs[..., 0]
        loss_actor_rew = -(ratio * rew_adv).mean()

        # compute safety loss
        values = [ratio * batch.advs[..., i]
                  for i in range(1, self.critics_num)] if self.use_lagrangian else []
        loss_actor_safety, stats_actor = self.safety_loss(values)

        rescaling = stats_actor["loss/rescaling"]
        loss_actor_total = rescaling * (loss_actor_rew + loss_actor_safety)

        stats_actor.update(
            {
                "loss/actor_rew": loss_actor_rew.item(),
                "loss/actor_total": loss_actor_total.item()
            }
        )
        return loss_actor_total, stats_actor

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:

        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # obtain the action distribution
                dist = self.forward(minibatch).dist
                # calculate policy loss
                loss_actor, stats_actor = self.policy_loss(minibatch, dist)

                flat_grads = self._get_flat_grad(
                    loss_actor, self.actor, retain_graph=True
                ).detach()

                # direction: calculate natural gradient
                with torch.no_grad():
                    old_dist = self(minibatch).dist

                kl = kl_divergence(old_dist, dist).mean()
                # calculate first order gradient of kl with respect to theta
                flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
                search_direction = -self._conjugate_gradients(
                    flat_grads, flat_kl_grad, nsteps=10
                )

                # stepsize: calculate max stepsize constrained by kl bound
                step_size = torch.sqrt(
                    2 * self._delta /
                    (search_direction *
                     self._MVP(search_direction, flat_kl_grad)).sum(0, keepdim=True)
                )

                # stepsize: linesearch stepsize
                with torch.no_grad():
                    flat_params = torch.cat(
                        [param.data.view(-1) for param in self.actor.parameters()]
                    )
                    for i in range(self._max_backtracks):
                        new_flat_params = flat_params + step_size * search_direction
                        self._set_from_flat_params(self.actor, new_flat_params)
                        # calculate kl and if in bound, loss actually down
                        new_dist = self(minibatch).dist
                        loss_actor_new, _ = self.policy_loss(minibatch, new_dist)
                        kl = kl_divergence(old_dist, new_dist).mean()

                        if kl < self._delta and loss_actor_new < loss_actor:
                            if i > 0:
                                self.logger.print(f"Backtracking to step {i}.")
                            break
                        elif i < self._max_backtracks - 1:
                            step_size = step_size * self._backtrack_coeff
                        else:
                            self._set_from_flat_params(self.actor, new_flat_params)
                            step_size = torch.tensor([0.0])
                            self.logger.print(
                                "Line search failed! It seems hyperparamters"
                                " are poor and need to be changed."
                            )

                ########################################
                for _ in range(self._optim_critic_iters):
                    loss_vf, stats_critic = self.critics_loss(minibatch)
                    self.optim.zero_grad()
                    loss_vf.backward()
                    self.optim.step()
                    self.gradient_steps += 1

                ent = dist.entropy().mean()
                self.logger.store(**stats_actor)
                self.logger.store(**stats_critic)
                self.logger.store(
                    kl=kl.item(),
                    step_size=step_size.item(),
                    entropy=ent.item(),
                    tab="loss"
                )

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(kl_v, self.actor,
                                                retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _conjugate_gradients(
        self,
        minibatch: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        x = torch.zeros_like(minibatch)
        r, p = minibatch.clone(), minibatch.clone()
        # Note: should be 'r, p = minibatch - MVP(x)', but for x=0, MVP(x)=0. Change if
        # doing warm start.
        rdotr = r.dot(r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _get_flat_grad(
        self, y: torch.Tensor, model: nn.Module, **kwargs: Any
    ) -> torch.Tensor:
        grads = torch.autograd.grad(y, model.parameters(), **kwargs)  # type: ignore
        return torch.cat([grad.reshape(-1) for grad in grads])

    def _set_from_flat_params(
        self, model: nn.Module, flat_params: torch.Tensor
    ) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size())
            )
            prev_ind += flat_size
        return model
