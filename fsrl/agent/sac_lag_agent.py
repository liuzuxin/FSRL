from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

from fsrl.agent import OffpolicyAgent
from fsrl.policy import SACLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic


class SACLagAgent(OffpolicyAgent):
    """Soft Actor-Critic (SAC) with PID Lagrangian agent.

    More details, please refer to https://arxiv.org/abs/1801.01290 (SAC) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param gym.Env env: The environment to train and evaluate the agent on.
    :param BaseLogger logger: A logger instance to log training and evaluation
        statistics, default to a dummy logger.
    :param float cost_limit: the constraint limit(s) for the Lagrangian optimization.
        (default: 10)
    :param str device: The device to use for training and inference, default to "cpu".
    :param int thread: The number of threads to use for training, ignored if `device` is
        "cuda", default to 4.
    :param int seed: The random seed for reproducibility, default to 10.
    :param float actor_lr: The learning rate of the actor network (default: 5e-4).
    :param float critic_lr: The learning rate of the critic network (default: 1e-3).
    :param Tuple[int, ...] hidden_sizes: The sizes of the hidden layers for the policy
        and value networks, default to (128, 128).
    :param bool auto_alpha: whether to automatically tune "alpha", the temperature.
        (default: True)
    :param float alpha_lr: the learning rate of learning "alpha" if ``auto_alpha`` is
        True. (default: 3e-4)
    :param float alpha: initial temperature for entropy regularization. (default: 0.005)
    :param float tau: target smoothing coefficient for soft update of target networks.
        (default: 0.05)
    :param int n_step: number of steps for multi-step learning. (default: 2)
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization.
        (default: True)
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization. (default: [0.05, 0.0005, 0.1])
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discount factor for future rewards. (default: 0.99)
    :param bool conditioned_sigma: Whether the variance of the Gaussian policy is
        conditioned on the state (default: True).
    :param bool unbounded: Whether the action space is unbounded. (default: False)
    :param bool last_layer_scale: Whether to scale the last layer output for the policy
        network. (default: False)
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. (default: True)
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. (default: True)
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). (default: "clip")
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. (default: None)


    .. seealso::

        Please refer to :class:`~fsrl.agent.BaseAgent` and
        :class:`~fsrl.agent.OffpolicyAgent` for more details of usage.
    """

    name = "SACLagAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        # general task params
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        # algorithm params
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        auto_alpha: bool = True,
        alpha_lr: float = 3e-4,
        alpha: float = 0.002,
        tau: float = 0.05,
        n_step: int = 2,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1),
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        conditioned_sigma: bool = True,
        unbounded: bool = True,
        last_layer_scale: bool = False,
        deterministic_eval: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    ) -> None:
        super().__init__()

        self.logger = logger
        self.cost_limit = cost_limit

        # set seed and computing
        seed_all(seed)
        torch.set_num_threads(thread)

        # model
        state_shape = env.observation_space.shape or env.observation_space.n
        action_shape = env.action_space.shape or env.action_space.n
        max_action = env.action_space.high[0]

        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = ActorProb(
            net,
            action_shape,
            max_action=max_action,
            device=device,
            conditioned_sigma=conditioned_sigma,
            unbounded=unbounded
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critics = []
        for _ in range(2):
            net1 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            net2 = Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            )
            critics.append(DoubleCritic(net1, net2, device=device).to(device))

        critic_optim = torch.optim.Adam(
            nn.ModuleList(critics).parameters(), lr=critic_lr
        )

        actor_critic = ActorCritic(actor, critics)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        if last_layer_scale:
            # do last policy layer scaling, this will make initial actions have (close
            # to) 0 mean and std, and will help boost performances, see
            # https://arxiv.org/abs/2006.05990, Fig.24 for details
            for m in actor.mu.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        if auto_alpha:
            target_entropy = -np.prod(env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACLagrangian(
            actor=actor,
            critics=critics,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            alpha=alpha,
            tau=tau,
            gamma=gamma,
            exploration_noise=None,
            n_step=n_step,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            reward_normalization=False,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )
