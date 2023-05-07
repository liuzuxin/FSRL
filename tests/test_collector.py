import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import MLP

from fsrl.data import BasicCollector, FastCollector, TrajectoryBuffer
from fsrl.policy import BasePolicy


class DummyPolicy(BasePolicy, torch.nn.Module):

    def __init__(self, action_space):
        super().__init__(MLP(1), MLP(1), action_space=action_space)

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.array([self.action_space.sample()]))

    def learn(self):
        pass


def test_basic_collector():
    env = gym.make("CartPole-v1")
    policy = DummyPolicy(env.action_space)
    buffer = TrajectoryBuffer(100)
    collector = BasicCollector(policy, env, traj_buffer=buffer)

    result = collector.collect(n_episode=1)
    assert result["n/ep"] == 1

    result = collector.collect(n_episode=1, random=True)
    assert result["n/ep"] == 1


def test_fast_collector():
    env = gym.make("CartPole-v1")
    policy = DummyPolicy(env.action_space)
    env = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(1)])
    collector = FastCollector(policy, env)

    result = collector.collect(n_episode=1)
    assert result["n/ep"] == 1

    result = collector.collect(n_episode=1, random=True)
    assert result["n/ep"] == 1


if __name__ == "__main__":
    test_fast_collector()
    test_basic_collector()
