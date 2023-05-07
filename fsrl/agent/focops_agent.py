from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from fsrl.agent import OnpolicyAgent
from fsrl.policy import FOCOPS
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic


class FOCOPSAgent(OnpolicyAgent):
    """First Order Constrained Optimization in Policy Space (FOCOPS) agent.

    More details, please refer to https://arxiv.org/pdf/2002.06506.pdf

    :param gym.Env env: The environment to train and evaluate the agent on.
    :param BaseLogger logger: A logger instance to log training and evaluation
        statistics, default to a dummy logger.
    :param float cost_limit: the constraint threshold. Default value is 10.
    :param str device: The device to use for training and inference, default to "cpu".
    :param int thread: The number of threads to use for training, ignored if `device` is
        "cuda", default to 4.
    :param int seed: The random seed for reproducibility, default to 10.
    :param float actor_lr: the learning rate of the actor network, default to 5e-4.
    :param float critic_lr: the learning rate of the critic network, default to 1e-3.
    :param Tuple[int, ...] hidden_sizes: The sizes of the hidden layers for the policy
        and value networks, default to (128, 128).
    :param bool unbounded: Whether the action space is unbounded, default to False.
    :param bool last_layer_scale: whether to scale the last layer output for the policy
        network, default to False.
    :param bool auto_nu: whether to automatically tune "nu", the cost coefficient.
        Default value is True.
    :param Union[float, Tuple[float, float, torch.Tensor]] nu: cost coefficient. It can
        also be a tuple representing [nu_max, nu_lr, nu]. Default value is 0.01.
    :param float nu_max: the max value of the cost coefficient if ``auto_nu`` is True.
        Default value is 2.
    :param float nu_lr: the learning rate of nu if ``auto_nu`` is True. Default value is
        0.01.
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
    :param Optional[torch.optim.lr_scheduler.LambdaLR] lr_scheduler: learning rate
        scheduler for the optimizer. Default value is None.

    .. seealso::

        Please refer to :class:`~fsrl.agent.BaseAgent` and
        :class:`~fsrl.agent.OnpolicyAgent` for more details of usage.
    """

    name = "FOCOPSAgent"

    def __init__(
        self,
        env: gym.Env,
        logger: BaseLogger = BaseLogger(),
        cost_limit: float = 10,
        device: str = "cpu",
        thread: int = 4,  # if use "cpu" to train
        seed: int = 10,
        actor_lr: float = 5e-4,
        critic_lr: float = 1e-3,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        unbounded: bool = False,
        last_layer_scale: bool = False,
        # FOCOPS specific arguments
        auto_nu: bool = True,
        nu: float = 0.01,
        nu_max: float = 2.0,
        nu_lr: float = 1e-2,
        l2_reg: float = 1e-3,
        delta: float = 0.02,
        eta: float = 0.02,
        tem_lambda: float = 0.95,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = 0.5,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        # Base policy common arguments
        gamma: float = 0.99,
        max_batchsize: int = 100000,
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

        actor = ActorProb(
            Net(state_shape, hidden_sizes=hidden_sizes, device=device),
            action_shape,
            max_action=max_action,
            unbounded=unbounded,
            device=device
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)

        critic = [
            Critic(
                Net(state_shape, hidden_sizes=hidden_sizes, device=device),
                device=device
            ).to(device) for _ in range(2)
        ]
        critic_optim = torch.optim.Adam(nn.ModuleList(critic).parameters(), lr=critic_lr)

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

        # replace DiagGuassian with Independent(Normal) which is equivalent pass *logits
        # to be consistent with policy.forward
        def dist(*logits):
            return Independent(Normal(*logits), 1)

        if auto_nu:
            nu = torch.zeros(1, requires_grad=False, device=device)
            nu = (nu_max, nu_lr, nu)

        self.policy = FOCOPS(
            actor,
            critic,
            actor_optim,
            critic_optim,
            dist,
            logger=logger,
            cost_limit=cost_limit,
            nu=nu,
            l2_reg=l2_reg,
            delta=delta,
            eta=eta,
            tem_lambda=tem_lambda,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            advantage_normalization=advantage_normalization,
            recompute_advantage=recompute_advantage,
            gamma=gamma,
            max_batchsize=max_batchsize,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_scheduler=lr_scheduler,
        )
