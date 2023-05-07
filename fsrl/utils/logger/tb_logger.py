import os.path as osp
from typing import Iterable, Tuple

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from fsrl.utils.logger.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    """A logger with tensorboard SummaryWriter to visualize and log statistics.

    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    :param str name: the experiment name. If None, it will use the current time as the
        name. Default to None.
    """

    def __init__(
        self, log_dir: str = None, log_txt: bool = True, name: str = None
    ) -> None:
        super().__init__(log_dir, log_txt, name)
        self.summary_writer = SummaryWriter(osp.join(self.log_dir, "tb"))

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
        """Writing data to the tf event file without resetting the current stored
        stats."""
        for k in self.logger_keys:
            self.summary_writer.add_scalar(k, self.get_mean(k), step)
        self.summary_writer.flush()

    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.
        If it finds nothing or an error occurs during the recover process, it will return
        the default parameters.

        :return Tuple[int, int, int]: episode, env_step, gradient_step.
        """
        ea = event_accumulator.EventAccumulator(self.summary_writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("update/episode")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("update/gradient_steps")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("update/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step
