from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from fsrl.data import FastCollector
from fsrl.policy import BasePolicy
from fsrl.trainer.base_trainer import BaseTrainer
from fsrl.utils import BaseLogger


class OnpolicyTrainer(BaseTrainer):
    """Create an iterator wrapper for on-policy training procedure.

    :param policy: an instance of the :class:`~fsrl.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing. If it's None, then
        no testing will be performed.
    :param int max_epoch: the maximum number of epochs for training. The training process
        might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int cost_limit: the constraint violation threshold.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int save_model_interval: how many epochs to save one checkpoint.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatedly in each epoch.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatedly in each epoch.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result, returns
        a boolean which indicates whether reaching the goal.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool show_progress: whether to display a progress bar when training. Default
        to True.

    .. note::

        We do not support the `step_per_collect` option as in Tianshou, \
            but only the `episode_per_collect` option for collecting data.
    """

    __doc__ = BaseTrainer.gen_doc("onpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
        self,
        policy: BasePolicy,
        train_collector: FastCollector,
        test_collector: Optional[FastCollector] = None,
        max_epoch: int = 10000,
        batch_size: int = 512,
        cost_limit: float = np.inf,
        step_per_epoch: int = 10000,
        repeat_per_collect: int = 4,
        episode_per_collect: int = 10,
        save_model_interval: int = 1,
        episode_per_test: Optional[int] = None,
        stop_fn: Optional[Callable[[float, float], bool]] = None,
        resume_from_log: bool = False,
        logger: BaseLogger = BaseLogger(),
        verbose: bool = True,
        show_progress: bool = True
    ):
        super().__init__(
            learning_type="onpolicy",
            policy=policy,
            max_epoch=max_epoch,
            batch_size=batch_size,
            train_collector=train_collector,
            cost_limit=cost_limit,
            test_collector=test_collector,
            step_per_epoch=step_per_epoch,
            repeat_per_collect=repeat_per_collect,
            save_model_interval=save_model_interval,
            episode_per_test=episode_per_test,
            episode_per_collect=episode_per_collect,
            stop_fn=stop_fn,
            resume_from_log=resume_from_log,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress
        )

    def policy_update_fn(self, stats_train: Dict[str, Any]) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        # Note, the first argument is 0: it will extract all the data from the buffer,
        # otherwise it will sample a batch with given sample_size.
        self.policy.pre_update_fn(
            stats_train=stats_train,
            batch_size=self.batch_size,
            buffer=self.train_collector.buffer
        )
        self.policy.update(
            0,
            self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )
        self.policy.post_update_fn(stats_train=stats_train)
        self.train_collector.reset_buffer(keep_statistics=True)
        # self.log_update_data(data, losses)


def onpolicy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OnpolicyTrainer run method.

    It is identical to ``OnpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OnpolicyTrainer(*args, **kwargs).run()


onpolicy_trainer_iter = OnpolicyTrainer
