from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.trainer.base_trainer import BaseTrainer
from fsrl.utils import BaseLogger


class OffpolicyTrainer(BaseTrainer):
    """Create an iterator wrapper for off-policy training procedure.

    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training process
        might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int cost_limit: the constraint violation threshold.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param float update_per_step: the number of times the policy network would be updated
        per transition after (step_per_collect) transitions are collected, e.g., if
        update_per_step set to 0.3, and step_per_collect is 256 , policy will be updated
        round(256 * 0.3 = 76.8) = 77 times after 256 transitions are collected by the
        collector.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param int save_model_interval: how many epochs to save one checkpoint.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result, returns
        a boolean which indicates whether reaching the goal.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training. Default
        to True.

    .. note::

        We do not support the `step_per_collect` option as in Tianshou, \
            but only the `episode_per_collect` option for collecting data.
    """

    __doc__ = BaseTrainer.gen_doc("offpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: FastCollector,
        test_collector: Optional[FastCollector] = None,
        max_epoch: int = 1000,
        batch_size: int = 512,
        cost_limit: float = np.inf,
        step_per_epoch: int = 10000,
        update_per_step: float = 0.1,
        episode_per_collect: int = 1,
        save_model_interval: int = 1,
        episode_per_test: Optional[int] = None,
        stop_fn: Optional[Callable[[float, float], bool]] = None,
        resume_from_log: bool = False,
        logger: BaseLogger = BaseLogger(),
        verbose: bool = True,
        show_progress: bool = True
    ):
        super().__init__(
            learning_type="offpolicy",
            policy=policy,
            max_epoch=max_epoch,
            batch_size=batch_size,
            train_collector=train_collector,
            cost_limit=cost_limit,
            test_collector=test_collector,
            step_per_epoch=step_per_epoch,
            update_per_step=update_per_step,
            save_model_interval=save_model_interval,
            episode_per_test=episode_per_test,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            resume_from_log=resume_from_log,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress
        )
        self.gradient_steps = 0

    def policy_update_fn(self, stats_train: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        self.policy.pre_update_fn(
            stats_train=stats_train,
            batch_size=self.batch_size,
            buffer=self.train_collector.buffer,
            update_per_step=self.update_per_step
        )
        for _ in range(round(self.update_per_step * stats_train["n/st"])):
            self.gradient_steps += 1
            self.policy.update(self.batch_size, self.train_collector.buffer)
        self.policy.post_update_fn(stats_train=stats_train)
        self.logger.store(gradient_steps=self.gradient_steps, tab="update")


def offpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OffpolicyTrainer(*args, **kwargs).run()


offpolicy_trainer_iter = OffpolicyTrainer
