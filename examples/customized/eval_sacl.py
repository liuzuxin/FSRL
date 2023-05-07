from dataclasses import asdict, dataclass

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
import torch
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb

from fsrl.data import FastCollector
from fsrl.policy import SACLagrangian
from fsrl.utils import BaseLogger
from fsrl.utils.exp_util import load_config_and_model, seed_all
from fsrl.utils.net.continuous import DoubleCritic


@dataclass
class EvalConfig:
    path: str = "logs"
    best: bool = False
    eval_episodes: int = 20
    worker: BaseVectorEnv = ShmemVectorEnv
    device: str = "cpu"
    render: bool = False


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)
    # seed
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["thread"])

    logger = BaseLogger()

    env = gym.make(cfg["task"])
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    # model
    net = Net(state_shape, hidden_sizes=cfg["hidden_sizes"], device=args.device)
    actor = ActorProb(
        net,
        action_shape,
        max_action=max_action,
        device=args.device,
        unbounded=cfg["unbounded"]
    ).to(args.device)

    critics = []
    for i in range(2):
        net1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=cfg["hidden_sizes"],
            concat=True,
            device=args.device
        )
        net2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=cfg["hidden_sizes"],
            concat=True,
            device=args.device
        )
        critics.append(DoubleCritic(net1, net2, device=args.device).to(args.device))

    policy = SACLagrangian(
        actor=actor,
        critics=critics,
        actor_optim=None,
        critic_optim=None,
        logger=logger,
        use_lagrangian=cfg["use_lagrangian"],
        action_space=env.action_space
    )
    policy.load_state_dict(model["model"])
    policy.eval()

    # collector
    test_envs = args.worker(
        [lambda: gym.make(cfg["task"]) for _ in range(args.eval_episodes)]
    )
    eval_collector = FastCollector(policy, test_envs)
    result = eval_collector.collect(n_episode=args.eval_episodes, render=args.render)
    rews, lens, cost = result["rew"], result["len"], result["cost"]
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()
