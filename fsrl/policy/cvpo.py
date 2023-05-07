import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from torch import nn
from torch.distributions import kl_divergence

from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger


class CVPO(BasePolicy):
    """Implementation of the Constrained Variational Policy Optimization (CVPO).

    More details, please refer to https://arxiv.org/abs/2201.11927.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~fsrl.policy.BasePolicy`. (s -> logits)
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param torch.optim.Optimizer actor_optim: the optimizer for the actor network.
    :param torch.optim.Optimizer critic_optim: the optimizer for the critic network(s).
    :param gym.Space action_space: the action space of the environment.
    :param Type[torch.distributions.Distribution] dist_fn: the probability distribution
        function for sampling actions.
    :param int max_episode_steps: the maximum number of steps per episode for computing
        the step-wise qc threshold.
    :param Optional[BaseLogger] logger: the logger instance for logging training
        information. (default=DummyLogger)
    :param Union[List, float] cost_limit: the constraint limit(s) for the optimization.
        (default=np.inf)
    :param float tau: target smoothing coefficient for soft update of target networks.
        (default=0.05)
    :param float gamma: the discount factor for future rewards. (default=0.99)
    :param int n_step: number of steps for multi-step learning. (default=2)
    :param int estep_iter_num: the number of iterations for the E-step. (default=1)
    :param float estep_kl: the KL divergence threshold for the E-step. (default=0.02)
    :param float estep_dual_max: the maximum value for the dual variable in the E-step.
        (default=20)
    :param float estep_dual_lr: the learning rate for the dual variable in the E-step.
        (default=0.02)
    :param int sample_act_num: the number of actions to sample for the E-step.
        (default=16)
    :param int mstep_iter_num: the number of iterations for the M-step. (default=1)
    :param float mstep_kl_mu: the KL divergence threshold for the M-step (mean).
        (default=0.005)
    :param float mstep_kl_std: the KL divergence threshold for the M-step (standard
        deviation). (default=0.0005)
    :param float mstep_dual_max: the maximum value for the dual variable in the M-step.
        (default=0.5)
    :param float mstep_dual_lr: the learning rate for the dual variable in the M-step.
        (default=0.1)
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. (default=True)
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. (default=True)
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). (default="clip")
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer.

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
        action_space: gym.Space,
        # CVPO specific arguments
        dist_fn: Type[torch.distributions.Distribution],
        max_episode_steps: int,
        logger: Optional[BaseLogger] = BaseLogger(),
        cost_limit: Union[List, float] = np.inf,
        tau: float = 0.05,
        gamma: float = 0.99,
        n_step: int = 2,
        # E-step
        estep_iter_num: int = 1,
        estep_kl: float = 0.02,
        estep_dual_max: float = 20,
        estep_dual_lr: float = 0.02,
        sample_act_num: int = 16,
        # M-step
        mstep_iter_num: int = 1,
        mstep_kl_mu: float = 0.005,
        mstep_kl_std: float = 0.0005,
        mstep_dual_max: float = 0.5,
        mstep_dual_lr: float = 0.1,
        # other param
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        super().__init__(
            actor=actor,
            critics=critics,
            dist_fn=dist_fn,
            logger=logger,
            gamma=gamma,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            action_space=action_space,
            lr_scheduler=lr_scheduler
        )

        self.actor_old = deepcopy(self.actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim
        self.critics_old = deepcopy(self.critics)
        self.critics_old.eval()
        self.critics_optim = critic_optim
        self.device = next(self.actor.parameters()).device
        self.dtype = next(self.actor.parameters()).dtype
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit
        # qc threshold in the E-step
        self.qc_thres = [
            c * (1 - self._gamma**max_episode_steps) / (1 - self._gamma) /
            max_episode_steps for c in self.cost_limit
        ]

        # E-step init
        self._estep_kl = estep_kl
        self._estep_iter_num = estep_iter_num
        self._estep_dual_max = estep_dual_max
        self._estep_dual_lr = estep_dual_lr
        self._sample_act_num = sample_act_num
        # the first dim is eta, others are lambda in the paper
        d = np.zeros(self.critics_num)
        d[0] = 1  # init eta to be 1
        self.estep_dual = torch.tensor(
            d, requires_grad=True, device=self.device, dtype=self.dtype
        )
        self.estep_optim = torch.optim.Adam([self.estep_dual], lr=self._estep_dual_lr)

        # M-step init
        self._mstep_kl_mu = mstep_kl_mu
        self._mstep_kl_std = mstep_kl_std
        self._mstep_iter_num = mstep_iter_num
        self._mstep_dual_max = mstep_dual_max
        self._mstep_dual_lr = mstep_dual_lr

        self._estep_duration = 0
        self._mstep_duration = 0

        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        self._discrete = True if self.action_type == "discrete" else False
        self._n_step = n_step
        self.__eps = np.finfo(np.float32).eps.item() * 10  # around 1e-6

    def update_cost_limit(self, cost_limit: float):
        """Update the cost limit threshold.

        :param float cost_limit: new cost threshold
        """
        self.cost_limit = [cost_limit] * (self.critics_num -
                                          1) if np.isscalar(cost_limit) else cost_limit

    def pre_update_fn(self, **kwarg: Any) -> Any:
        """Init the mstep optimizer and dual variables."""
        self.mstep_dual_mu = torch.zeros(
            1, requires_grad=True, device=self.device, dtype=self.dtype
        )
        self.mstep_dual_std = torch.zeros(
            1, requires_grad=True, device=self.device, dtype=self.dtype
        )
        self.mstep_optim = torch.optim.Adam(
            [self.mstep_dual_mu, self.mstep_dual_std], lr=self._mstep_dual_lr
        )

    def post_update_fn(self, **kwarg: Any) -> Any:
        """Update the old actor network."""
        with torch.no_grad():
            self.actor_old.load_state_dict(self.actor.state_dict())

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
        obs_next_result = self(batch, model="actor", input='obs_next')
        act = obs_next_result.act
        target_q_list = []
        for i in range(self.critics_num):
            target_q, _ = self.critics_old[i].predict(batch.obs_next, act)
            target_q_list.append(target_q)
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
        model: str = "actor",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        logits, hidden = model(obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def critics_loss(
        self, batch: Batch, critics: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, dict]:
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
            for j in range(len(current_q_list)):
                td = current_q_list[j].flatten() - target_q
                td_average += td
                loss_i += (td.pow(2) * weight).mean()

            loss_critic += loss_i
            stats_critic["loss/loss_q" + str(i)] = loss_i.item()
            stats_critic["estep/val_q" + str(i)] = torch.mean(target_q).item()
            if i >= 1:
                stats_critic["estep/thres_q" + str(i)] = self.qc_thres[i - 1]
        optimizer.zero_grad()
        loss_critic.backward()
        optimizer.step()
        td_average /= self.critics_num * 2
        stats_critic["loss/q_total"] = loss_critic.item()
        return td_average, stats_critic

    def _estep_dual_loss(self, q_values):
        eta = self.estep_dual[0]
        K = q_values[0].shape[-1]
        loss = eta * self._estep_kl
        combined_q = q_values[0].detach()  # (B, K)
        for i in range(1, self.critics_num):
            combined_q -= self.estep_dual[i] * q_values[i].detach()
            loss += self.estep_dual[i] * self.qc_thres[i - 1]
        loss += eta * torch.mean(torch.logsumexp(combined_q / eta, dim=1) - np.log(K))
        return loss

    @staticmethod
    def gaussian_kl(
        mu_old: torch.Tensor, std_old: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoupled KL between two multivariate Gaussians with diagonal covariance.

        See https://arxiv.org/pdf/1812.02256.pdf Sec. 4.2.1 for details. kl_mu = KL(
        pi(mu_old, std_old) || pi(mu, std_old) ) kl_std = KL( pi(mu_old, std_old) ||
        pi(mu_old, std) )

        :param mu_old: (B, n)
        :param mu: (B, n)
        :param std_old: (B, n)
        :param std: (B, n)
        :return: kl_mu, kl_std: scalar mean and covariance terms of the KL
        """
        var_old, var = std_old**2, std**2
        # for numerical stability
        var_old = torch.clamp_min(var_old, 1e-6)
        var = torch.clamp_min(var, 1e-6)

        # note, this kl's demoninator is the old var rather than the new var
        kl_mu = 0.5 * (mu_old - mu)**2 / var_old
        kl_mu = torch.sum(kl_mu, dim=-1).mean()

        kl_std = 0.5 * (torch.log(var / var_old) + var_old / var - 1)
        kl_std = torch.sum(kl_std, dim=-1).mean()  # Sum over the dimensions

        return kl_mu, kl_std

    def policy_loss(self, batch: Batch, **kwarg):

        # E-step begin
        t_start = time.time()
        obs = torch.as_tensor(
            batch.obs, device=self.device, dtype=torch.float32
        )  # (B, ds)
        # for continuous action space, sample K particles
        K = self._sample_act_num
        B = obs.shape[0]
        da = batch.act.shape[-1]
        ds = obs.shape[-1]
        with torch.no_grad():
            old_result = self(batch, model="actor_old", input="obs")
            old_dist = old_result.dist  # (B, da)
            sample_act = old_dist.sample((K, ))  # (K, B, da)
            expanded_obs = obs[None, ...].expand(K, -1, -1)  # (K, B, ds)
            q_values = []
            # TODO, use critics old or the current?
            for i in range(self.critics_num):
                target_q, _ = self.critics[i].predict(
                    expanded_obs.reshape(-1, ds), sample_act.reshape(-1, da)
                )
                target_q = target_q.reshape(K, B)
                q_values.append(target_q.T)  # (critic_num, B, K)

        # optimize
        for _ in range(self._estep_iter_num):
            self.estep_optim.zero_grad()
            estep_loss = self._estep_dual_loss(q_values)
            estep_loss.backward()
            self.estep_optim.step()
            self.logger.store(tab="loss", estep_loss=estep_loss.item())
        self.estep_dual.data.clamp_(min=self.__eps, max=self._estep_dual_max)
        # detach the estep dual variable for M-step
        estep_dual = []
        for i in range(self.critics_num):
            estep_dual.append(self.estep_dual[i].detach().item())
            self.logger.store(**{"estep/dual" + str(i): estep_dual[i]})

        # compute the optimal non-parametric variational distribution
        optimal_q = q_values[0].T  # (K, B)
        for i in range(1, self.critics_num):
            optimal_q -= estep_dual[i] * q_values[i].T
        optimal_q = torch.softmax((optimal_q) / estep_dual[0], dim=0).detach()  # (K, B)

        t_estep = time.time()
        self._estep_duration += t_estep - t_start
        self.logger.store(tab="estep", estep_time=self._estep_duration)

        # M-step begin

        mu_old, std_old = old_result.logits
        mu_old, std_old = mu_old.detach(), std_old.detach()
        for _ in range(self._mstep_iter_num):
            result = self(batch, model="actor", input="obs")

            # MLE loss
            mu, std = result.logits
            dist1 = self.dist_fn(mu, std_old)
            dist2 = self.dist_fn(mu_old, std)
            likelihood = dist1.expand((K, B)).log_prob(sample_act) + dist2.expand(
                (K, B)
            ).log_prob(sample_act)  # (K, B)
            loss_mle = -torch.mean(optimal_q * likelihood)

            # update dual variables to regularize the KL
            kl_mu, kl_std = self.gaussian_kl(mu_old, std_old, mu, std)
            mstep_dual_loss = self.mstep_dual_mu * (self._mstep_kl_mu - kl_mu).detach(
            ) + self.mstep_dual_std * (self._mstep_kl_std - kl_std).detach()
            self.mstep_optim.zero_grad()
            mstep_dual_loss.backward()
            self.mstep_optim.step()

            # KL loss
            dual_mu = np.clip(self.mstep_dual_mu.item(), 0.0, self._mstep_dual_max)
            dual_std = np.clip(self.mstep_dual_std.item(), 0.0, self._mstep_dual_max)
            loss_kl = dual_mu * (kl_mu - self._mstep_kl_mu
                                 ) + dual_std * (kl_std - self._mstep_kl_std)

            loss_actor = loss_mle + loss_kl

            # optimize the policy network
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            entropy = torch.mean(dist1.entropy() + dist2.entropy()).item()

            self.logger.store(
                tab="mstep",
                mstep_kl_mu=kl_mu.item(),
                mstep_kl_std=kl_std.item(),
                mstep_loss_kl=loss_kl.item(),
                mstep_loss_mle=loss_mle.item(),
                mstep_loss_total=loss_actor.item(),
                mstep_dual_mu=dual_mu,
                mstep_dual_std=dual_std,
                entropy=entropy
            )
        self._mstep_duration += time.time() - t_estep
        self.logger.store(tab="mstep", mstep_time=self._mstep_duration)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic
        td, stats_critic = self.critics_loss(batch, self.critics, self.critics_optim)
        batch.weight = td  # prio-buffer
        # actor
        self.policy_loss(batch)

        self.sync_weight()
        self.logger.store(**stats_critic)

    def get_extra_state(self):
        """Save the dual variables and their optimizers.

        This function is called when call the policy.state_dict(), see
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_extra_state
        """
        # if len(self.lag_optims): return [optim.state_dict() for optim in
        #     self.lag_optims] else: return None

    def set_extra_state(self, state):
        """Load the dual variables and their optimizers.

        This function is called from load_state_dict() to handle any extra state found
        within the state_dict.
        """
