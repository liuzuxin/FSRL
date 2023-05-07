from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.env import BaseVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import TRPOLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class TRPOLagAgent(OnpolicyAgent):
    """Trust Region Policy Optimization (TRPO) with PID Lagrangian agent.

    More details, please refer to https://arxiv.org/abs/1502.05477 (TRPO) and
    https://arxiv.org/abs/2007.03964 (PID Lagrangian).

    :param gym.Env env: The environment to train and evaluate the agent on.
    :param BaseLogger logger: A logger instance to log training and evaluation
        statistics, default to a dummy logger.
    :param float cost_limit: the constraint limit(s) for the Lagrangian optimization
        (default: 10).
    :param str device: The device to use for training and inference, default to "cpu".
    :param int thread: The number of threads to use for training, ignored if `device` is
        "cuda", default to 4.
    :param int seed: The random seed for reproducibility, default to 10.
    :param float lr: The learning rate, default to 5e-4.
    :param float target_kl: the target KL divergence for the line search (default:
        0.001).
    :param Tuple[int, ...] hidden_sizes: The sizes of the hidden layers for the policy
        and value networks, default to (128, 128).
    :param bool unbounded: Whether the action space is unbounded, default to False.
    :param bool last_layer_scale: Whether to scale the last layer output for the policy
        network, default to False.
    :param float backtrack_coeff: the coefficient for backtracking during the line search
        (default: 0.8).
    :param int max_backtracks: the maximum number of backtracks allowed during the line
        search (default: 10).
    :param int optim_critic_iters: the number of optimization iterations for the critic
        network (default: 20).
    :param float gae_lambda: the GAE lambda value (default: 0.95).
    :param bool advantage_normalization: whether to normalize advantage (default: True).
    :param bool use_lagrangian: whether to use the Lagrangian constraint optimization
        (default: True).
    :param List lagrangian_pid: the PID coefficients for the Lagrangian constraint
        optimization (default: [0.05, 0.0005, 0.1]).
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
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer (default: None).

    .. seealso::

        Please refer to :class:`~fsrl.agent.BaseAgent` and
        :class:`~fsrl.agent.OnpolicyAgent` for more details of usage.
    """

    name = "TRPOLagAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        lr: float = 5e-4,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        unbounded: bool = False,
        last_layer_scale: bool = False,
        # PPO specific arguments
        target_kl: float = 0.001,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        optim_critic_iters: int = 20,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
        # Lagrangian specific arguments
        use_lagrangian: bool = True,
        lagrangian_pid: Tuple = (0.05, 0.0005, 0.1),
        rescaling: bool = True,
        # Base policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 99999,
        reward_normalization: bool = False,  # can decrease final perf
        deterministic_eval: bool = True,
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
            net, action_shape, max_action=max_action, unbounded=unbounded, device=device
        ).to(device)
        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device) for _ in range(2)
        ]

        torch.nn.init.constant_(actor.sigma_param, -0.5)
        actor_critic = ActorCritic(actor, critic)
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
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        # replace DiagGuassian with Independent(Normal) which is equivalent
        # pass *logits to be consistent with policy.forward
        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = TRPOLagrangian(
            actor,
            critic,
            optim,
            dist,
            logger=logger,
            # PPO specific arguments
            target_kl=target_kl,
            backtrack_coeff=backtrack_coeff,
            max_backtracks=max_backtracks,
            optim_critic_iters=optim_critic_iters,
            gae_lambda=gae_lambda,
            advantage_normalization=advantage_normalization,
            # Lagrangian specific arguments
            use_lagrangian=use_lagrangian,
            lagrangian_pid=lagrangian_pid,
            cost_limit=cost_limit,
            rescaling=rescaling,
            # Base policy common arguments
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler
        )

    def learn(
        self,
        train_envs: Union[gym.Env, BaseVectorEnv],
        test_envs: Union[gym.Env, BaseVectorEnv] = None,
        epoch: int = 300,
        episode_per_collect: int = 20,
        step_per_epoch: int = 10000,
        repeat_per_collect: int = 4,
        buffer_size: int = 100000,
        testing_num: int = 2,
        batch_size: int = 99999,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """See :meth:`~fsrl.agent.OnpolicyAgent.learn` for details."""
        return super().learn(
            train_envs, test_envs, epoch, episode_per_collect, step_per_epoch,
            repeat_per_collect, buffer_size, testing_num, batch_size, reward_threshold,
            save_interval, resume, save_ckpt, verbose, show_progress
        )
