import uuid
from typing import Iterable

import wandb

from fsrl.utils.logger.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """Weights and Biases logger that sends data to https://wandb.ai/.

    A typical usage example: ::

        config = {...} project = "test_cvpo" group = "SafetyCarCircle-v0" name =
        "default_param" log_dir = "logs"

        logger = WandbLogger(config, project, group, name, log_dir)
        logger.save_config(config)

        agent = CVPOAgent(env, logger=logger) agent.learn(train_envs)

    :param str config: experiment configurations. Default to an empty dict.
    :param str project: W&B project name. Default to "fsrl".
    :param str group: W&B group name. Default to "test".
    :param str name: W&B experiment run name. If None, it will use the current time as
        the name. Default to None.
    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    """

    def __init__(
        self,
        config: dict = {},
        project: str = "fsrl",
        group: str = "test",
        name: str = None,
        log_dir: str = "log",
        log_txt: bool = True
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.wandb_run = wandb.init(
            project=project,
            group=group,
            name=name,
            id=str(uuid.uuid4()),
            resume="allow",
            config=config,  # type: ignore
        ) if not wandb.run else wandb.run
        # wandb.run.save()

    def write(
        self,
        step: int,
        display: bool = True,
        display_keys: Iterable[str] = None
    ) -> None:
        """Writing data to somewhere and reset the stored data.

        :param int step: the current training step or epochs
        :param bool display: whether print the logged data in terminal, default to False
        :param Iterable[str] display_keys: a list of keys to be printed. If None, print
            all stored keys, default to None.
        """
        self.store(tab="update", env_step=step)
        self.write_without_reset(step)
        return super().write(step, display, display_keys)

    def write_without_reset(self, step: int) -> None:
        """Sending data to wandb without resetting the current stored stats."""
        wandb.log(self.stats_mean, step=step)

    def restore_data(self) -> None:
        """Not implemented yet"""
