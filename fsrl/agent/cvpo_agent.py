from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb
from torch.distributions import Independent, Normal

from fsrl.agent import OffpolicyAgent
from fsrl.policy import CVPO
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic, SingleCritic


class CVPOAgent(OffpolicyAgent):
    """Constrained Variational Policy Optimization (CVPO) agent.

    More details, please refer to https://arxiv.org/abs/2201.11927.

    :param gym.Env env: The Gym environment to train the agent on.
    :param BaseLogger logger: The logger to use for the agent (default is DummyLogger).
    :param int cost_limit: The cost limit of the task.
    :param str device: The device to use for training (default is 'cpu').
    :param str thread: The number of threads to use for training when using the CPU
        (default is 4).
    :param int seed: The random seed to use for training (default is 10).
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
    :param float actor_lr: The learning rate of the actor network (default is 5e-4).
    :param float critic_lr: The learning rate of the critic network (default is 1e-3).
    :param float gamma: The discount factor (default is 0.98).
    :param int n_step: The number of steps to look ahead when computing returns (default
        is 2).
    :param float tau: The critics soft update coefficient (default is 0.05).
    :param Tuple[int, ...] hidden_sizes: The sizes of the hidden layers in the actor and
        critic networks (default is (128, 128)).
    :param bool double_critic: Whether to use two critic networks instead of one (default
        is False).
    :param bool conditioned_sigma: Whether the variance of the Gaussian policy is
        conditioned on the state (default is True).
    :param bool unbounded: Whether to use an unbounded output layer for the actor network
        (default is False).
    :param bool last_layer_scale: Whether to scale the last layer of the actor network
        (default is False).
    :param bool deterministic_eval: Whether to use a deterministic policy during
        evaluation (default is True).
    :param bool action_scaling: Whether to scale actions by the maximum action value
        (default is True).
    :param str action_bound_method: The method to use for action bounds ('clip' or
        'tanh') (default is 'clip').
    :param torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler (default is
        None).

    .. seealso::

        Please refer to :class:`~fsrl.agent.BaseAgent` and
        :class:`~fsrl.agent.OffpolicyAgent` for more details of usage.
    """

    name = "CVPOAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        # general task params
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        # CVPO arguments,
        estep_iter_num: int = 1,
        estep_kl: float = 0.02,
        estep_dual_max: float = 20,
        estep_dual_lr: float = 0.02,
        sample_act_num: int = 16,
        mstep_iter_num: int = 1,
        mstep_kl_mu: float = 0.005,
        mstep_kl_std: float = 0.0005,
        mstep_dual_max: float = 0.5,
        mstep_dual_lr: float = 0.1,
        # other algorithm params,
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.98,
        n_step: int = 2,
        tau: float = 0.05,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        double_critic: bool = False,
        conditioned_sigma: bool = True,
        unbounded: bool = False,
        last_layer_scale: bool = False,
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

        assert hasattr(
            env.spec, "max_episode_steps"
        ), "Please use an env wrapper to provide 'max_episode_steps' for CVPO"

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
            if double_critic:
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
            else:
                net_c = Net(
                    state_shape,
                    action_shape,
                    hidden_sizes=hidden_sizes,
                    concat=True,
                    device=device
                )
                critics.append(SingleCritic(net_c, device=device).to(device))

        critic_optim = torch.optim.Adam(
            nn.ModuleList(critics).parameters(), lr=critic_lr
        )
        if not conditioned_sigma:
            torch.nn.init.constant_(actor.sigma_param, -0.5)
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

        def dist(*logits):
            return Independent(Normal(*logits), 1)

        self.policy = CVPO(
            actor=actor,
            critics=critics,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            logger=logger,
            action_space=env.action_space,
            dist_fn=dist,
            max_episode_steps=env.spec.max_episode_steps,
            cost_limit=cost_limit,
            tau=tau,
            gamma=gamma,
            n_step=n_step,
            estep_iter_num=estep_iter_num,
            estep_kl=estep_kl,
            estep_dual_max=estep_dual_max,
            estep_dual_lr=estep_dual_lr,
            sample_act_num=sample_act_num,  # for continous action space
            mstep_iter_num=mstep_iter_num,
            mstep_kl_mu=mstep_kl_mu,
            mstep_kl_std=mstep_kl_std,
            mstep_dual_max=mstep_dual_max,
            mstep_dual_lr=mstep_dual_lr,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler
        )
