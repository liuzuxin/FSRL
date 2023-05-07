from argparse import Action
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from torch import nn
from torch.distributions import kl_divergence

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic


class CPO(BasePolicy):
    """Implementation of the Constrained Policy Optimization (CPO).

    More details, please refer to https://arxiv.org/abs/1705.10528.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param Type[torch.distributions.Distribution] dist_fn: the distribution function for
        the policy.
    :param BaseLogger logger: the logger instance for logging training information.
    :param float target_kl: the target KL divergence for the line search. (default: 0.01)
    :param float backtrack_coeff: the coefficient for backtracking during the line
        search. (default: 0.8)
    :param float damping_coeff: the damping coefficient for the Fisher matrix. (default:
        0.1)
    :param int max_backtracks: the maximum number of backtracks allowed during the line
        search. (default: 10)
    :param int optim_critic_iters: the number of optimization iterations for the critic
        network. (default: 20)
    :param float l2_reg: the L2 regularization coefficient for the critic network.
        (default: 0.001)
    :param float gae_lambda: the GAE lambda value. (default: 0.95)
    :param bool advantage_normalization: normalize advantage if True. (default: True)
    :param Union[List, float] cost_limit: the constraint limit(s) for the Lagrangian
        optimization. (default: np.inf)
    :param float gamma: the discount factor for future rewards. (default: 0.99)
    :param int max_batchsize: the maximum batch size for updating the policy. (default:
        99999)
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

        Please refer to :class:`~fsrl.policy.BasePolicy` for more detailed hyperparameter
        explanations and usage.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        logger: BaseLogger = BaseLogger(),
        # CPO specific arguments
        target_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        damping_coeff: float = 0.1,
        max_backtracks: int = 10,
        optim_critic_iters: int = 20,
        l2_reg: float = 0.001,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        cost_limit: Union[List, float] = np.inf,
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
            actor, critics, dist_fn, logger, gamma, max_batchsize, reward_normalization,
            deterministic_eval, action_scaling, action_bound_method, observation_space,
            action_space, lr_scheduler
        )
        self.optim = optim
        self._cost_limit = cost_limit
        self._lambda = gae_lambda
        self._norm_adv = advantage_normalization
        self._max_backtracks = max_backtracks
        self._optim_critic_iters = optim_critic_iters
        self._l2_reg = l2_reg
        self._delta = target_kl
        self._backtrack_coeff = backtrack_coeff
        self._damping_coeff = damping_coeff

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
        batch = self.compute_gae_returns(batch, buffer, indices, self._lambda)
        if self._norm_adv:
            for i in range(self.critics_num):
                adv = batch.advs[..., i]
                mean, std = adv.mean(), adv.std()
                batch.advs[..., i] = (adv - mean) / std
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

    def critics_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:
        critic_losses = torch.zeros(1)
        stats = {}
        for i, critic in enumerate(self.critics):
            value = critic(minibatch.obs).flatten()
            ret = minibatch.rets[..., i]
            vf_loss = (ret - value).pow(2).mean()
            for param in critic.parameters():
                vf_loss += param.pow(2).sum() * self._l2_reg
            critic_losses += vf_loss
            stats["loss/vf" + str(i)] = vf_loss.item()
        self.optim.zero_grad()
        critic_losses.backward()
        self.optim.step()
        stats["loss/vf_total"] = critic_losses.item()
        return critic_losses, stats

    def _get_objective(
        self, logp: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.exp(logp - logp_old) * adv)

    def _get_cost_surrogate(
        self, logp: torch.Tensor, logp_old: torch.Tensor, cadv: torch.Tensor
    ) -> torch.Tensor:
        cost_surrogate = self._ave_cost_return + torch.mean(
            torch.exp(logp - logp_old) * cadv
        ) - torch.mean(cadv)
        return cost_surrogate

    def _MVP(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = torch.dot(flat_kl_grad, v)
        flat_kl_grad_grad = self._get_flat_grad(kl_v, self.actor, retain_graph=True)
        return flat_kl_grad_grad + v * self._damping_coeff

    def _conjugate_gradients(
        self,
        g: torch.Tensor,
        flat_kl_grad: torch.Tensor,
        nsteps: int = 10,
        residual_tol: float = 1e-8
    ) -> torch.Tensor:
        x = torch.zeros_like(g)
        r, p = g.clone(), g.clone()
        rs_old = torch.sum(r * r)
        for _ in range(nsteps):
            z = self._MVP(p, flat_kl_grad)
            alpha = rs_old / torch.sum(p * z)
            x += alpha * p
            r -= alpha * z
            rs_new = torch.sum(r * r)
            if rs_new < residual_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def _get_flat_grad(
        self,
        y: torch.Tensor,
        model: nn.Module,
        retain_graph: bool = False,
        create_graph: bool = False
    ) -> torch.Tensor:
        retain_graph = True if create_graph else retain_graph
        grads = torch.autograd.grad(
            y,
            model.parameters(),  # type: ignore
            retain_graph=retain_graph,
            create_graph=create_graph
        )
        return torch.cat([grad.view(-1) for grad in grads])

    def _get_flat_params(self, model: nn.Module) -> torch.Tensor:
        flat_params = torch.cat([p.view(-1) for p in model.parameters()])
        return flat_params

    def _set_from_flat_params(self, model: nn.Module, new_params: torch.Tensor) -> None:
        n = 0
        for param in model.parameters():
            numel = param.numel()
            new_param = new_params[n:n + numel].view(param.size())
            param.data = new_param
            n += numel

    def policy_loss(self, minibatch: Batch) -> Tuple[torch.Tensor, dict]:

        self.actor.train()
        # get objective & KL & cost surrogate
        dist = self.forward(minibatch).dist
        ent = dist.entropy().mean()
        logp = dist.log_prob(minibatch.act)

        dist_old = self.dist_fn(*(minibatch.mean_old, minibatch.std_old))  # type: ignore
        kl = kl_divergence(dist_old, dist).mean()

        objective = self._get_objective(logp, minibatch.logp_old, minibatch.advs[..., 0])
        cost_surrogate = self._get_cost_surrogate(
            logp, minibatch.logp_old, minibatch.advs[..., 1]
        )
        loss_actor_total = objective + cost_surrogate

        # get gradient
        grad_g = self._get_flat_grad(objective, self.actor, retain_graph=True)
        grad_b = self._get_flat_grad(-cost_surrogate, self.actor, retain_graph=True)
        flat_kl_grad = self._get_flat_grad(kl, self.actor, create_graph=True)
        H_inv_g = self._conjugate_gradients(grad_g, flat_kl_grad)
        approx_g = self._MVP(H_inv_g, flat_kl_grad)
        c_value = cost_surrogate - self._cost_limit

        # solve Lagrangian problem
        EPS = 1e-8
        if torch.dot(grad_b, grad_b) <= EPS and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = [
                torch.zeros(1) for _ in range(5)
            ]
            scalar_q = torch.dot(approx_g, H_inv_g)
            optim_case = 4
        else:
            H_inv_b = self._conjugate_gradients(grad_b, flat_kl_grad)
            approx_b = self._MVP(H_inv_b, flat_kl_grad)
            scalar_q = torch.dot(approx_g, H_inv_g)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)

            # should be always positive (Cauchy-Shwarz)
            A_value = scalar_q - scalar_r**2 / scalar_s
            # does safety boundary intersect trust region? (positive = yes)
            B_value = 2 * self._delta - c_value**2 / scalar_s
            if c_value < 0 and B_value < 0:
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else:
                optim_case = 0

        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q / (2 * self._delta))
            nu = torch.zeros_like(lam)
        elif optim_case in [1, 2]:
            LA, LB = [0, scalar_r / c_value], [scalar_r / c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(A_value / B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q / (2 * self._delta)), LB)
            f_a = lambda lam: -0.5 * (A_value / (lam + EPS) + B_value * lam
                                      ) - scalar_r * c_value / (scalar_s + EPS)
            f_b = lambda lam: -0.5 * (scalar_q / (lam + EPS) + 2 * self._delta * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            lam = torch.tensor(lam)
            nu = max(0, (lam * c_value - scalar_r).item()) / (scalar_s + EPS)
        else:
            nu = torch.sqrt(2 * self._delta / (scalar_s + EPS))
            lam = torch.zeros_like(nu)
        # line search
        with torch.no_grad():
            delta_theta = (1. / (lam + EPS)) * (
                H_inv_g + nu * H_inv_b
            ) if optim_case > 0 else nu * H_inv_b
            delta_theta /= torch.norm(delta_theta)
            beta = 1.0
            # sometimes the scalar_q can be negative causing lam to be nan
            if not torch.isnan(lam):
                init_theta = self._get_flat_params(self.actor).clone().detach()
                init_objective = objective.clone().detach()
                init_cost_surrogate = cost_surrogate.clone().detach()
                for _ in range(self._max_backtracks):
                    theta = beta * delta_theta + init_theta
                    self._set_from_flat_params(self.actor, theta)
                    dist = self.forward(minibatch).dist
                    logp = dist.log_prob(minibatch.act)
                    new_kl = kl_divergence(dist_old, dist).mean().item()
                    new_objective = self._get_objective(
                        logp, minibatch.logp_old, minibatch.advs[..., 0]
                    )
                    new_cost_surrogate = self._get_cost_surrogate(
                        logp, minibatch.logp_old, minibatch.advs[..., 1]
                    )
                    if new_kl <= self._delta and \
                        (new_objective > init_objective if optim_case > 1 else True) and \
                        new_cost_surrogate - init_cost_surrogate <= max(-c_value.item(), 0): # noqa
                        break
                    beta *= self._backtrack_coeff

        stats_actor = {
            "loss/kl": kl.item(),
            "loss/entropy": ent.item(),
            "loss/rew_loss": objective.item(),
            "loss/cost_loss": cost_surrogate.item(),
            "loss/optim_A": A_value.item(),
            "loss/optim_B": B_value.item(),
            "loss/optim_C": c_value.item(),
            "loss/optim_Q": scalar_q.item(),
            "loss/optim_R": scalar_r.item(),
            "loss/optim_S": scalar_s.item(),
            "loss/optim_lam": lam.item(),
            "loss/optim_nu": nu.item(),
            "loss/optim_case": optim_case,
            "loss/step_size": beta
        }
        return loss_actor_total, stats_actor

    def learn(  # type: ignore
            self, batch: Batch, batch_size: int, repeat: int,
            **kwargs: Any) -> Dict[str, List[float]]:

        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):

                for _ in range(self._optim_critic_iters):
                    loss_vf, stats_critic = self.critics_loss(minibatch)

                # calculate policy loss
                loss_actor, stats_actor = self.policy_loss(minibatch)

                self.gradient_steps += 1
                self.logger.store(**stats_actor)
                self.logger.store(**stats_critic)

        self.logger.store(gradient_steps=self.gradient_steps, tab="update")
