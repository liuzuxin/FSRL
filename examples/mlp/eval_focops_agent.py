import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import bullet_safety_gym

try:
    import safety_gymnasium
except ImportError:
    print("safety_gymnasium is not found.")
import gymnasium as gym
import pyrallis
from tianshou.env import BaseVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.agent import FOCOPSAgent
from fsrl.utils import BaseLogger, TensorboardLogger, WandbLogger
from fsrl.utils.exp_util import auto_name, load_config_and_model, seed_all


@dataclass
class EvalConfig:
    path: str = "logs"
    best: bool = False
    eval_episodes: int = 20
    parallel_eval: bool = True
    device: str = "cpu"
    render: bool = False
    train_mode: bool = False


@pyrallis.wrap()
def eval(args: EvalConfig):
    cfg, model = load_config_and_model(args.path, args.best)

    task = cfg["task"]
    demo_env = gym.make(task)

    agent = FOCOPSAgent(
        env=demo_env,
        logger=BaseLogger(),
        device=args.device,
        use_lagrangian=cfg["use_lagrangian"],
        thread=cfg["thread"],
        seed=cfg["seed"],
        hidden_sizes=cfg["hidden_sizes"],
        unbounded=cfg["unbounded"],
        last_layer_scale=cfg["last_layer_scale"],
    )

    if args.parallel_eval:
        assert args.render is False, "please use single env when rendering"
        test_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(args.eval_episodes)]
        )
    else:
        render_mode = "human" if args.render else None
        test_envs = gym.make(task, render_mode=render_mode)

    rews, lens, cost = agent.evaluate(
        test_envs=test_envs,
        state_dict=model["model"],
        eval_episodes=args.eval_episodes,
        render=args.render,
        train_mode=args.train_mode
    )
    print("Traing mode: ", args.train_mode)
    print(f"Eval reward: {rews}, cost: {cost}, length: {lens}")


if __name__ == "__main__":
    eval()
