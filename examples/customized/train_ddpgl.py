import os
import pprint
from dataclasses import asdict

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
import torch
import torch.nn as nn
from tianshou.data import VectorReplayBuffer
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from fsrl.config.ddpgl_cfg import (
    Bullet1MCfg,
    Bullet5MCfg,
    Bullet10MCfg,
    Mujoco2MCfg,
    Mujoco10MCfg,
    Mujoco20MCfg,
    MujocoBaseCfg,
    TrainCfg,
)
from fsrl.data import FastCollector
from fsrl.policy import DDPGLagrangian
from fsrl.trainer import OffpolicyTrainer
from fsrl.utils import TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, seed_all
from fsrl.utils.net.common import ActorCritic
from fsrl.utils.net.continuous import DoubleCritic

TASK_TO_CFG = {
    # bullet safety gym tasks
    "SafetyCarRun-v0": Bullet1MCfg,
    "SafetyBallRun-v0": Bullet1MCfg,
    "SafetyBallCircle-v0": Bullet1MCfg,
    "SafetyCarCircle-v0": TrainCfg,
    "SafetyDroneRun-v0": TrainCfg,
    "SafetyAntRun-v0": TrainCfg,
    "SafetyDroneCircle-v0": Bullet5MCfg,
    "SafetyAntCircle-v0": Bullet10MCfg,
    # safety gymnasium tasks
    "SafetyPointCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle1Gymnasium-v0": Mujoco2MCfg,
    "SafetyCarCircle2Gymnasium-v0": Mujoco2MCfg,
    "SafetyPointGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyPointPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarGoal2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarButton2Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush1Gymnasium-v0": MujocoBaseCfg,
    "SafetyCarPush2Gymnasium-v0": MujocoBaseCfg,
    "SafetyHalfCheetahVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyHopperVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetySwimmerVelocityGymnasium-v1": MujocoBaseCfg,
    "SafetyWalker2dVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyAntVelocityGymnasium-v1": Mujoco10MCfg,
    "SafetyHumanoidVelocityGymnasium-v1": Mujoco20MCfg,
}


@pyrallis.wrap()
def train(args: TrainCfg):
    # set seed and computing
    seed_all(args.seed)
    torch.set_num_threads(args.thread)

    task = args.task
    default_cfg = TASK_TO_CFG[task]() if task in TASK_TO_CFG else TrainCfg()
    # use the default configs instead of the input args.
    if args.use_default_cfg:
        default_cfg.task = args.task
        default_cfg.seed = args.seed
        default_cfg.device = args.device
        default_cfg.logdir = args.logdir
        default_cfg.project = args.project
        default_cfg.group = args.group
        default_cfg.suffix = args.suffix
        args = default_cfg

    # setup logger
    cfg = asdict(args)
    default_cfg = asdict(default_cfg)
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.project, args.group)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    training_num = min(args.training_num, args.episode_per_collect)
    worker = eval(args.worker)
    train_envs = worker([lambda: gym.make(args.task) for _ in range(training_num)])
    test_envs = worker([lambda: gym.make(args.task) for _ in range(args.testing_num)])

    # model
    env = gym.make(args.task)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    net = Net(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, action_shape, max_action=max_action,
                  device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    nets = [
        Net(
            state_shape,
            action_shape,
            hidden_sizes=args.hidden_sizes,
            concat=True,
            device=args.device
        ) for i in range(2)
    ]
    critic = [Critic(n, device=args.device).to(args.device) for n in nets]
    critic_optim = torch.optim.Adam(
        nn.ModuleList(critic).parameters(), lr=args.critic_lr
    )

    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    policy = DDPGLagrangian(
        actor=actor,
        critics=critic,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        tau=args.tau,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        n_step=args.n_step,
        use_lagrangian=args.use_lagrangian,
        lagrangian_pid=args.lagrangian_pid,
        cost_limit=args.cost_limit,
        rescaling=args.rescaling,
        gamma=args.gamma,
        reward_normalization=False,
        deterministic_eval=args.deterministic_eval,
        action_scaling=args.action_scaling,
        action_bound_method=args.action_bound_method,
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_scheduler=None
    )

    # collector
    train_collector = FastCollector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = FastCollector(policy, test_envs)

    def stop_fn(reward, cost):
        return reward > args.reward_threshold and cost < args.cost_limit

    def checkpoint_fn():
        return {"model": policy.state_dict()}

    if args.save_ckpt:
        logger.setup_checkpoint_fn(checkpoint_fn)

    # trainer
    trainer = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        cost_limit=args.cost_limit,
        step_per_epoch=args.step_per_epoch,
        update_per_step=args.update_per_step,
        episode_per_test=args.testing_num,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_model_interval=args.save_interval,
        verbose=args.verbose,
    )

    for epoch, epoch_stat, info in trainer:
        logger.store(tab="train", cost_limit=args.cost_limit)
        print(f"Epoch: {epoch}")
        print(info)

    if __name__ == "__main__":
        pprint.pprint(info)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final eval reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")

        policy.train()
        collector = FastCollector(policy, env)
        result = collector.collect(n_episode=10, render=args.render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        print(f"Final train reward: {rews.mean()}, cost: {cost}, length: {lens.mean()}")


if __name__ == "__main__":
    train()
