from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from fsrl.agent import OffpolicyAgent
from fsrl.policy import DDPGLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class DDPGLagAgent(OffpolicyAgent):
    """Deep Deterministic Policy Gradient (DDPG) with PID Lagrangian agent.

    More details, please refer to https://arxiv.org/abs/1509.02971 (DDPG) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param gym.Env env: The environment to train and evaluate the agent on.
    :param BaseLogger logger: A logger instance to log training and evaluation
        statistics, default to a dummy logger.
    :param float cost_limit: The maximum constraint cost allowed, default to 10.
    :param str device: The device to use for training and inference, default to "cpu".
    :param int thread: The number of threads to use for training, ignored if `device` is
        "cuda", default to 4.
    :param int seed: The random seed for reproducibility, default to 10.
    :param float actor_lr: The learning rate of the actor network (default is 5e-4).
    :param float critic_lr: The learning rate of the critic network (default is 1e-3).
    :param Tuple[int, ...] hidden_sizes: The sizes of the hidden layers in the actor and
        critic networks (default is (128, 128)).
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
    :param bool rescaling: whether use the rescaling trick for Lagrangian multiplier, see
        Alg. 1 in http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
    :param float gamma: the discount factor for future rewards. Default is 0.99.
    :param bool deterministic_eval: whether to use deterministic action selection during
        evaluation. Default is True.
    :param bool action_scaling: whether to scale the actions according to the action
        space bounds. Default is True.
    :param str action_bound_method: the method for handling actions that exceed the
        action space bounds ("clip" or other custom methods). Default is "clip".
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. Default is None.

    .. seealso::

        Please refer to :class:`~fsrl.agent.BaseAgent` and
        :class:`~fsrl.agent.OffpolicyAgent` for more details of usage.
    """

    name = "DDPGLagAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        # general task params
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        # algorithm params
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        n_step: int = 3,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple[float, ...] = (0.5, 0.001, 0.1),
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
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
        actor = Actor(net, action_shape, max_action=max_action, device=device).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        nets = [
            Net(
                state_shape,
                action_shape,
                hidden_sizes=hidden_sizes,
                concat=True,
                device=device
            ) for i in range(2)
        ]
        critic = [Critic(n, device=device).to(device) for n in nets]
        critic_optim = torch.optim.Adam(nn.ModuleList(critic).parameters(), lr=critic_lr)

        actor_critic = ActorCritic(actor, critic)
        # orthogonal initialization
        for m in actor_critic.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        self.policy = DDPGLagrangian(
            actor=actor,
            critics=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            tau=tau,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            n_step=n_step,
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            gamma=gamma,
            reward_normalization=False,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )
