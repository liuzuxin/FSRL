import bullet_safety_gym
import gymnasium as gym
from tianshou.env import DummyVectorEnv

from fsrl.agent import (
    CPOAgent,
    CVPOAgent,
    DDPGLagAgent,
    FOCOPSAgent,
    PPOLagAgent,
    SACLagAgent,
    TRPOLagAgent,
)


def test_all_agents():
    task = "SafetyBallRun-v0"
    training_num = 1
    train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(1)])

    agent_list = [
        TRPOLagAgent, PPOLagAgent, CPOAgent, CVPOAgent, SACLagAgent, DDPGLagAgent,
        FOCOPSAgent
    ]

    ########### test all the agents ###########
    total_time = 0
    for agent in agent_list:
        agent: TRPOLagAgent
        agent = agent(gym.make(task), cost_limit=9999)
        print(f"Testing {agent.name}....")
        epoch, stats, info = agent.learn(
            train_envs, test_envs, 10, 1, 1000, verbose=False, show_progress=True
        )
        duration, reward, cost = info["duration"], info["best_reward"], info["best_cost"]
        total_time += duration
        print(
            f"{agent.name} uses {duration:.2f}s, \
            {epoch} epochs to reach reward: {reward} and cost: {cost}"
        )
        assert epoch < 10, f"{agent.name} didnot finish training \
             in 10 epochs, check the implementation."

    rew, len, cost = agent.evaluate(test_envs)
    assert rew > 450, f"{agent.name} didnot finish training \
        in 10 epochs, check the implementation."

    print(f"Total time usage: {total_time:.2f}s")


if __name__ == "__main__":
    test_all_agents()
