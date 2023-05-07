import time
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, to_numpy

from fsrl.data.traj_buf import TrajectoryBuffer
from fsrl.policy import BasePolicy


class BasicCollector:
    """A basic collector for a single environment.

    This collector doesn't support vector env and is served as experimental purpose. It
    supports to store collected data in the :class:`~fsrl.data.TrajectoryBuffer` with a
    grid filter, which can be used to memory-efficiently collect trajectory-wise
    interaction dataset.

    Example of data saving: ::

        traj_buffer = TrajectoryBuffer(max_traj_num) collector = BasicCollector(policy,
        env, traj_buffer=traj_buffer) collector.collect(n_episodes)

        traj_buffer.save(logdir)

    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class. If set
        to None, it will not store the data. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified with
        corresponding policy's exploration noise. If so, "policy. exploration_noise(act,
        batch)" will be called automatically to add the exploration noise into action.
        Default to False.
    :param TrajectoryBuffer traj_buffer: the buffer used to store trajectories

    .. note::

        Please make sure the given environment has a time limitation (can be done), \
            because we only support the `n_episode` collect option.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: gym.Env,
        buffer: Optional[ReplayBuffer] = None,
        exploration_noise: Optional[bool] = False,
        traj_buffer: Optional[TrajectoryBuffer] = None,
    ):
        self.env = env
        self.policy = policy
        if buffer is None:
            buffer = ReplayBuffer(1)

        self.buffer = buffer
        self.exploration_noise = exploration_noise
        self._action_space = self.env.action_space

        self.traj_buffer = traj_buffer
        self.reset(False)

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached to
            the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing convert empty
        # Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            act={},
            rew={},
            cost={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval,
                                  (tuple, list
                                   )) and len(rval) == 2 and isinstance(rval[1], dict)
        if returns_info:
            obs, info = rval
            self.data.info = [info]
        else:
            obs = rval
        self.data.obs = [obs]

    def collect(
        self,
        n_episode: int = 0,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default to
            False.
        :param float render: the sleep time between rendering consecutive frames. Default
            to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            We don not support the `n_step` collection method in Tianshou, because using
            `n_episode` only can facilitate the episodic cost computation and better
            evaluate the agent.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``total_cost`` cumulative costs in this collect.
            * ``cost`` mean of episodic costs.
            * ``truncated`` mean of episodic truncation.
            * ``terminated`` mean of episodic termination.
        """
        start_time = time.time()

        step_count = 0
        total_cost = 0
        termination_count = 0
        truncation_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []

        while True:
            # get the next action
            if random:
                act_sample = self._action_space.sample()
                act_sample = self.policy.map_action_inverse(act_sample)
                self.data.update(act=[act_sample])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        result = self.policy(
                            Batch(obs=self.data.obs, info=self.data.info)
                        )

                else:
                    result = self.policy(Batch(obs=self.data.obs, info=self.data.info))

                act = to_numpy(result.act)[0]
                # print(act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(act=[act])

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = np.squeeze(self.policy.map_action(self.data.act))
            # print(action_remap.shape) print("Env action space shape: ",
            # np.shape(self.env.action_space.sample())) step in env
            result = self.env.step(action_remap)
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            cost = info.get("cost", 0)

            self.data.update(
                obs_next=[obs_next],
                rew=[rew],
                terminated=[terminated],
                truncated=[truncated],
                done=[done],
                cost=[cost],
                info=[info]
            )

            termination_count += terminated
            truncation_count += truncated

            total_cost += cost

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, 1)

            if self.traj_buffer is not None:
                traj_data = Batch(
                    observations=self.data.obs,
                    next_observations=self.data.obs_next,
                    actions=[action_remap],
                    rewards=self.data.rew,
                    costs=self.data.cost,
                    terminals=self.data.terminated,
                    timeouts=self.data.truncated
                )
                self.traj_buffer.store(traj_data)

            step_count += 1

            if done:
                episode_count += 1
                episode_lens.append(ep_len)
                episode_rews.append(ep_rew)
                # now we copy obs_next to obs, but since there might be finished
                # episodes, we have to reset finished envs first.
                self.reset_env(gym_reset_kwargs)

            self.data.obs = self.data.obs_next

            if episode_count >= n_episode:
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        self.reset_env()

        done_count = truncation_count + termination_count

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rew": np.mean(episode_rews),
            "len": np.mean(episode_lens),
            "total_cost": total_cost,
            "cost": total_cost / episode_count,
            "truncated": truncation_count / done_count,
            "terminated": termination_count / done_count,
        }
