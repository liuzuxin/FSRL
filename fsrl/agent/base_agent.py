from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.env import BaseVectorEnv, DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.trainer import OffpolicyTrainer, OnpolicyTrainer
from fsrl.utils import BaseLogger


class BaseAgent(ABC):
    """The base class for a default agent.

    A agent class should have the following parts:

    * :meth:`~fsrl.agent.BaseAgent.__init__`: initialize the agent, including the policy,
        networks, optimizers, and so on;
    * :meth:`~fsrl.agent.BaseAgent.learn`: start training given the learning parameters;
    * :meth:`~fsrl.agent.BaseAgent.evaluate`: evaluate the agent multiple episodes;
    * :attr:`~fsrl.agent.BaseAgent.state_dict`: the agent state dictionary that can be
        saved as checkpoints;

    Example of usage: ::

        # initialize the CVPO agent
        agent = CVPOAgent(env, other_algo_params) # train multiple epochs
        agent.learn(training_envs, other_training_params)

        # test after the training is finished agent.eval(testing_envs)

        # test with agent's state_dict agent.eval(testing_envs, agent.state_dict)

    All of the agent classes must inherit :class:`~fsrl.agent.BaseAgent`.
    """

    name = "BaseAgent"

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        self.policy: BasePolicy
        self.task = None
        self.logger = BaseLogger()
        self.cost_limit = 0

    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """Train the policy on a set of training environments."""
        raise NotImplementedError

    def evaluate(
        self,
        test_envs: Union[gym.Env, BaseVectorEnv],
        state_dict: Optional[dict] = None,
        eval_episodes: int = 10,
        render: bool = False,
        train_mode: bool = False
    ) -> Tuple[float, float, float]:
        """Evaluate the policy on a set of test environments.

        :param Union[gym.Env, BaseVectorEnv] test_envs: A single environment or a
            vectorized environment to evaluate the policy on.
        :param Optional[dict] state_dict: An optional dictionary containing the state
             params of the agent to be evaluated., defaults to None
        :param int eval_episodes: The number of episodes to evaluate, defaults to 10
        :param bool render: Whether to render the environment during evaluation, defaults
            to False
        :param bool train_mode: Whether to set the policy to training mode during
            evaluation, defaults to False
        :return Tuple: rewards, episode lengths, and constraint costs obtained during
            evaluation.
        """
        if state_dict is not None:
            self.policy.load_state_dict(state_dict)
        if train_mode:
            self.policy.train()
        else:
            self.policy.eval()

        eval_collector = FastCollector(self.policy, test_envs)
        result = eval_collector.collect(n_episode=eval_episodes, render=render)
        rews, lens, cost = result["rew"], result["len"], result["cost"]
        # term, trun = result["terminated"], result["truncated"] print(f"Termination:
        # {term}, truncation: {trun}") print(f"Eval reward: {rews.mean()}, cost: {cost},
        # length: {lens.mean()}")
        return rews, lens, cost

    @property
    def state_dict(self):
        """Return the policy's state_dict."""
        return self.policy.state_dict()


class OffpolicyAgent(BaseAgent):
    """The base class for an off-policy agent.

    The :meth:`~srl.agent.OffpolicyAgent.learn`: function is customized to work with the
    off-policy trainer. See :class:`~fsrl.agent.BaseAgent` for more details.
    """

    name = "OffpolicyAgent"

    def __init__(self) -> None:
        super().__init__()

    def learn(
        self,
        train_envs: Union[gym.Env, BaseVectorEnv],
        test_envs: Union[gym.Env, BaseVectorEnv] = None,
        epoch: int = 300,
        episode_per_collect: int = 5,
        step_per_epoch: int = 3000,
        update_per_step: float = 0.1,
        buffer_size: int = 100000,
        testing_num: int = 2,
        batch_size: int = 256,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,  # TODO
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """Train the policy on a set of training environments.

        :param Union[gym.Env, BaseVectorEnv] train_envs: A single environment or a
            vectorized environment to train the policy on.
        :param Union[gym.Env, BaseVectorEnv] test_envs: A single environment or a
            vectorized environment to evaluate the policy on, default to None.
        :param int epoch: The number of training epochs, defaults to 300.
        :param int episode_per_collect: The number of episodes to collect before each
            policy update, defaults to 5.
        :param int step_per_epoch: The number of environment steps per epoch, defaults to
            3000.
        :param float update_per_step: The ratio of policy updates to environment steps, \
            defaults to 0.1.
        :param int buffer_size: The maximum size of the replay buffer, defaults to
            100000.
        :param int testing_num: The number of episodes to use for evaluation, defaults to
            2.
        :param int batch_size: The batch size for each policy update, defaults to 256.
        :param float reward_threshold: The reward threshold for early stopping, \
            defaults to 450.
        :param int save_interval: The interval (in epochs) for saving the policy model,
            defaults to 4.
        :param bool resume: Whether to resume training from the last checkpoint, defaults
            to False.
        :param bool save_ckpt: Whether to save the policy model, defaults to True.
        :param bool verbose: Whether to print progress information during training,
            defaults to True.
        :param bool show_progress: Whether to show the tqdm training progress bar,
            defaults to True
        """
        assert self.policy is not None, "The policy is not initialized"
        # set policy to train mode
        self.policy.train()
        # collector
        if isinstance(train_envs, gym.Env):
            buffer = ReplayBuffer(buffer_size)
        else:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        train_collector = FastCollector(
            self.policy,
            train_envs,
            buffer,
            exploration_noise=True,
        )

        test_collector = FastCollector(
            self.policy, test_envs
        ) if test_envs is not None else None

        def stop_fn(reward, cost):
            return reward > reward_threshold and cost < self.cost_limit

        def checkpoint_fn():
            return {"model": self.state_dict}

        if save_ckpt:
            self.logger.setup_checkpoint_fn(checkpoint_fn)

        # trainer
        trainer = OffpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            batch_size=batch_size,
            cost_limit=self.cost_limit,
            step_per_epoch=step_per_epoch,
            update_per_step=update_per_step,
            episode_per_test=testing_num,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            logger=self.logger,
            resume_from_log=resume,
            save_model_interval=save_interval,
            verbose=verbose,
            show_progress=show_progress
        )

        for epoch, _epoch_stat, info in trainer:
            self.logger.store(tab="train", cost_limit=self.cost_limit)
            if verbose:
                print(f"Epoch: {epoch}", info)

        return epoch, _epoch_stat, info


class OnpolicyAgent(BaseAgent):
    """The base class for an on-policy agent.

    The :meth:`~srl.agent.OnpolicyAgent.learn`: function is customized to work with the \
        on-policy trainer.
    See :class:`~fsrl.agent.BaseAgent` for more details.
    """

    name = "OnpolicyAgent"

    def __init__(self) -> None:
        super().__init__()

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
        batch_size: int = 512,
        reward_threshold: float = 450,
        save_interval: int = 4,
        resume: bool = False,
        save_ckpt: bool = True,
        verbose: bool = True,
        show_progress: bool = True
    ) -> None:
        """Train the policy on a set of training environments.

        :param Union[gym.Env, BaseVectorEnv] train_envs: A single environment or a
            vectorized environment to train the policy on.
        :param Union[gym.Env, BaseVectorEnv] test_envs: A single environment or a
            vectorized environment to evaluate the policy on, defaults to None.
        :param int epoch: The number of training epochs, defaults to 300
        :param int episode_per_collect: The number of episodes collected per data
            collection, defaults to 20
        :param int step_per_epoch: The number of steps per training epoch, defaults to
            10000
        :param int repeat_per_collect: The number of repeats of policy update for one
            episode collection, defaults to 4
        :param int buffer_size: The size of the replay buffer, defaults to 100000
        :param int testing_num: The number of episodes to evaluate during testing,
            defaults to 2
        :param int batch_size: The batch size for training, default is 99999 for
            :class:`~fsrl.agent.TRPOLagAgent` :class:`~fsrl.agent.CPOLagAgent`, and is
            512 for others
        :param float reward_threshold: The threshold for stopping training when the mean
            reward exceeds it, defaults to 450
        :param int save_interval: The number of epochs to save the policy, defaults to 4
        :param bool resume: Whether to resume training from the saved checkpoint,
            defaults to False
        :param bool save_ckpt: Whether to save the policy model, defaults to True
        :param bool verbose: Whether to print the training information, defaults to True
        :param bool show_progress: Whether to show the tqdm training progress bar,
            defaults to True
        """
        assert self.policy is not None, "The policy is not initialized"
        # set policy to train mode
        self.policy.train()
        # collector
        if isinstance(train_envs, gym.Env):
            buffer = ReplayBuffer(buffer_size)
        else:
            buffer = VectorReplayBuffer(buffer_size, len(train_envs))
        train_collector = FastCollector(
            self.policy,
            train_envs,
            buffer,
            exploration_noise=True,
        )
        test_collector = FastCollector(
            self.policy, test_envs
        ) if test_envs is not None else None

        def stop_fn(reward, cost):
            return reward > reward_threshold and cost < self.cost_limit

        def checkpoint_fn():
            return {"model": self.state_dict}

        if save_ckpt:
            self.logger.setup_checkpoint_fn(checkpoint_fn)

        # trainer
        trainer = OnpolicyTrainer(
            policy=self.policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epoch,
            batch_size=batch_size,
            cost_limit=self.cost_limit,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            episode_per_test=testing_num,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            logger=self.logger,
            resume_from_log=resume,
            save_model_interval=save_interval,
            verbose=verbose,
            show_progress=show_progress
        )

        for epoch, _epoch_stat, info in trainer:
            self.logger.store(tab="train", cost_limit=self.cost_limit)
            if verbose:
                print(f"Epoch: {epoch}", info)

        return epoch, _epoch_stat, info
