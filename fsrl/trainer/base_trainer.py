import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union

import numpy as np
import tqdm
from tianshou.utils import DummyTqdm, MovAvg, deprecation, tqdm_config

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.utils import BaseLogger


class BaseTrainer(ABC):
    """An iterator base class for trainers procedure.

    Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results on
    every epoch. The usage of the trainer is almost identical with tianshou's trainer,
    but with some modifications of the parameters.

    :param learning_type str: type of learning iterator, available choices are
        "offpolicy" and "onpolicy", we don't support "offline" yet.
    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param train_collector: the collector used for training.
    :param test_collector: the collector used for testing. If it's None, then no testing
        will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training process
        might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int cost_limit: the constraint violation threshold.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data twice.
        (on-policy method)
    :param float update_per_step: the number of gradient steps per env_step (off-policy).
    :param float save_model_interval: how many epochs to save one checkpoint.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param function stop_fn: a function with signature ``f(reward, cost) ->
        bool``, receives the average undiscounted returns of the testing result, returns
        a boolean which indicates whether reaching the goal.
    :param bool resume_from_log: resume env_step and other metadata from existing
        tensorboard log. Default to False.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print tabular information. Default to True.
    :param bool show_progress: whether to display a progress bar when training. Default
        to True.
    """

    @staticmethod
    def gen_doc(learning_type: str) -> str:
        """Document string for subclass trainer."""
        step_means = f'The "step" in {learning_type} trainer means '
        if learning_type != "offline":
            step_means += "an environment step (a.k.a. transition)."
        else:  # offline
            step_means += "a gradient step."

        trainer_name = learning_type.capitalize() + "Trainer"

        return f"""An iterator class for {learning_type} trainer procedure.

        Returns an iterator that yields a 3-tuple (epoch, stats, info) of train results
        on every epoch.

        {step_means}

        Example usage:

        ::

            trainer = {trainer_name}(...) for epoch, epoch_stat, info in trainer:
                print("Epoch:", epoch) print(epoch_stat) print(info)
                do_something_with_policy() query_something_about_policy()
                make_a_plot_with(epoch_stat) display(info)

        - epoch int: the epoch number
        - epoch_stat dict: a large collection of metrics of the current epoch
        - info dict: result returned from :func:`~fsrl.BaseTrainer.gather_update_info`

        You can even iterate on several trainers at the same time:

        ::

            trainer1 = {trainer_name}(...) trainer2 = {trainer_name}(...) for result1,
            result2, ... in zip(trainer1, trainer2, ...):
                compare_results(result1, result2, ...)
        """

    def __init__(
        self,
        learning_type: str,
        policy: BasePolicy,
        train_collector: FastCollector,
        test_collector: Optional[FastCollector] = None,
        max_epoch: int = 100,
        batch_size: int = 512,
        cost_limit: float = np.inf,
        step_per_epoch: Optional[int] = None,
        repeat_per_collect: Optional[int] = None,
        update_per_step: Union[int, float] = 1,
        save_model_interval: int = 1,
        episode_per_test: Optional[int] = None,
        episode_per_collect: int = 1,
        stop_fn: Optional[Callable[[float, float], bool]] = None,
        resume_from_log: bool = False,
        logger: BaseLogger = BaseLogger(),
        verbose: bool = True,
        show_progress: bool = True,
    ):

        self.policy = policy
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.logger = logger
        self.cost_limit = cost_limit

        self.start_time = time.time()

        # used for determining stopping criterio.
        self.stats_smoother: DefaultDict[str, MovAvg] = defaultdict(MovAvg)
        # The best performance is dertemined by [reward, feasibility status]. If two
        # policies are both (in)feasible, the higher reward one is better. Otherwise, the
        # feasible one is better.
        self.best_perf_rew = -np.inf
        self.best_perf_cost = np.inf
        self.start_epoch = 0
        self.env_step = 0
        self.cum_cost = 0
        self.cum_episode = 0
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.episode_per_collect = episode_per_collect
        self.episode_per_test = episode_per_test

        self.update_per_step = update_per_step
        self.save_model_interval = save_model_interval
        self.repeat_per_collect = repeat_per_collect

        self.batch_size = batch_size

        self.stop_fn = stop_fn

        self.verbose = verbose
        self.show_progress = show_progress
        self.resume_from_log = resume_from_log

        self.epoch = self.start_epoch
        self.best_epoch = self.start_epoch
        self.stop_fn_flag = False

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.env_step = 0
        # TODO
        # if self.resume_from_log:
        #     self.start_epoch, self.env_step = \
        #         self.logger.restore_data()
        #     self.best_epoch = self.start_epoch

        self.start_time = time.time()

        self.train_collector.reset_stat()

        if self.test_collector is not None:
            assert self.episode_per_test is not None
            self.test_collector.reset_stat()

        self.epoch = self.start_epoch
        self.stop_fn_flag = False

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Tuple[int, Dict, Dict]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        # iterator exhaustion check
        if self.epoch > self.max_epoch:
            raise StopIteration

        # exit flag 1, when stop_fn succeeds in train_step or test_step
        if self.stop_fn_flag:
            raise StopIteration

        # set policy in train mode (not training the policy)
        self.policy.train()

        progress = tqdm.tqdm if self.show_progress else DummyTqdm
        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:

                stats_train = self.train_step()
                t.update(stats_train["n/st"])

                self.policy_update_fn(stats_train)

                t.set_postfix(
                    cost=stats_train["cost"],
                    rew=stats_train["rew"],
                    length=stats_train["len"]
                )

                self.logger.write_without_reset(self.env_step)

        # test
        if self.test_collector is not None:
            self.test_step()

        # train and test collector info
        update_info = self.gather_update_info()
        self.logger.store(tab="update", **update_info)

        if self.epoch % self.save_model_interval == 0:
            self.logger.save_checkpoint()

        if self.perf_is_better(test=True):
            self.logger.save_checkpoint(suffix="best")

        if self.stop_fn and self.stop_fn(self.best_perf_rew, self.best_perf_cost):
            self.stop_fn_flag = True
            self.logger.print("Early stop due to the stop_fn met.", "red")

        epoch_stats = self.logger.stats_mean

        # after write, all the stats will be resetted.
        self.logger.write(self.env_step, display=self.verbose)

        update_info.update(
            {
                "best_reward": self.best_perf_rew,
                "best_cost": self.best_perf_cost
            }
        )

        return self.epoch, epoch_stats, update_info

    def perf_is_better(self, test: bool = True) -> bool:
        # use the testing or the training metric to determine the best
        mode = "test" if test and self.test_collector is not None else "train"
        rew = self.logger.get_mean(mode + "/reward")
        cost = self.logger.get_mean(mode + "/cost")
        if self.best_perf_cost > self.cost_limit:
            if cost <= self.cost_limit or rew > self.best_perf_rew:
                self.best_perf_cost = cost
                self.best_perf_rew = rew
                return True
        else:
            if cost <= self.cost_limit and rew > self.best_perf_rew:
                self.best_perf_cost = cost
                self.best_perf_rew = rew
                return True
        return False

    def test_step(self) -> Dict[str, Any]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        self.test_collector.reset_env()
        self.test_collector.reset_buffer()
        self.policy.eval()
        stats_test = self.test_collector.collect(n_episode=self.episode_per_test)

        self.logger.store(
            **{
                "test/reward": stats_test["rew"],
                "test/cost": stats_test["cost"],
                "test/length": int(stats_test["len"]),
            }
        )
        return stats_test

    def train_step(self) -> Dict[str, Any]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        stats_train = self.train_collector.collect(self.episode_per_collect)

        self.env_step += int(stats_train["n/st"])
        self.cum_cost += stats_train["total_cost"]
        self.cum_episode += int(stats_train["n/ep"])
        self.logger.store(
            **{
                "update/episode": self.cum_episode,
                "update/cum_cost": self.cum_cost,
                "train/reward": stats_train["rew"],
                "train/cost": stats_train["cost"],
                "train/length": int(stats_train["len"]),
            }
        )
        return stats_train

    @abstractmethod
    def policy_update_fn(self, result: Dict[str, Any]) -> None:
        """Policy update function for different trainer implementation.

        :param result: collector's return value.
        """

    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed (feed
        the entire iterator into a zero-length deque).
        """
        deque(self, maxlen=0)
        return self.gather_update_info()

    def gather_update_info(self) -> Dict[str, Any]:
        """A simple wrapper of gathering information from collectors.

        :return: A dictionary with the following keys:

            * ``train_collector_time`` the time (s) for collecting transitions in the \
                training collector;
            * ``train_model_time`` the time (s) for training models;
            * ``train_speed`` the speed of training (env_step per second);
            * ``test_time`` the time (s) for testing;
            * ``test_speed`` the speed of testing (env_step per second);
            * ``duration`` the total elapsed time (s).
        """
        duration = max(0, time.time() - self.start_time)
        model_time = max(0, duration - self.train_collector.collect_time)
        result = {"duration": duration}
        if self.test_collector is not None:
            collect_test = self.test_collector.collect_time
            model_time = max(0, model_time - collect_test)

            test_speed = self.test_collector.collect_step / collect_test
            result.update(
                {
                    "test_time": collect_test,
                    "test_speed": test_speed,
                    "duration": duration,
                }
            )
            train_speed = self.train_collector.collect_step / (duration - collect_test)
        else:
            train_speed = self.train_collector.collect_step / duration
        result.update(
            {
                "train_collector_time": self.train_collector.collect_time,
                "train_model_time": model_time,
                "train_speed": train_speed,
                "remaining_epoch": self.max_epoch - self.epoch
            }
        )
        return result
