import atexit
import json
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
import yaml

from fsrl.utils.logger.logger_util import RunningAverage, colorize, convert_json


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer.  All the loggers
    create four panels by default: `train`, `test`, `loss`, and `update`.  Try to
    overwrite write() method to customize your own logger.

    :param str log_dir: the log directory. Default to None.
    :param bool log_txt: whether to log data in ``log_dir`` with name ``progress.txt``.
        Default to True.
    :param str name: the experiment name. If None, it will use the current time as the
        name. Default to None.
    """

    def __init__(self, log_dir=None, log_txt=True, name=None) -> None:
        super().__init__()
        self.name = name if name is not None else time.strftime("%Y-%m-%d_exp")
        self.log_dir = osp.join(log_dir, name) if log_dir is not None else None
        self.log_fname = "progress.txt"
        if log_dir:
            if osp.exists(self.log_dir):
                warning_msg = colorize(
                    "Warning: Log dir %s already exists! Some logs may be overwritten." %
                    self.log_dir, "magenta", True
                )
                print(warning_msg)
            else:
                os.makedirs(self.log_dir)
            if log_txt:
                self.output_file = open(osp.join(self.log_dir, self.log_fname), 'w')
                atexit.register(self.output_file.close)
                print(
                    colorize(
                        "Logging data to %s" % self.output_file.name, 'green', True
                    )
                )
        else:
            self.output_file = None
        self.first_row = True
        self.checkpoint_fn = None
        self.reset_data()

    def setup_checkpoint_fn(self, checkpoint_fn: Optional[Callable] = None) -> None:
        """Setup the function to obtain the model checkpoint, it will be called \
            when using ```logger.save_checkpoint()```.

        :param Optional[Callable] checkpoint_fn: the hook function to get the \
            checkpoint dictionary, defaults to None.
        """
        self.checkpoint_fn = checkpoint_fn

    def reset_data(self) -> None:
        """Reset stored data"""
        self.log_data = defaultdict(RunningAverage)

    def store(self, tab: str = None, **kwargs) -> None:
        """Store any values to the current epoch buffer with prefix `tab/`.

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs) logger.save_config(locals())

        :param str tab: the prefix of the logging data, defaults to None.
        """
        for k, v in kwargs.items():
            if tab is not None:
                k = tab + "/" + k
            self.log_data[k].add(np.mean(v))

    def write(
        self,
        step: int,
        display: bool = False,
        display_keys: Iterable[str] = None
    ) -> None:
        """Writing data to somewhere and reset the stored data.

        :param int step: the current training step or epochs
        :param bool display: whether print the logged data in terminal, default to False
        :param Iterable[str] display_keys: a list of keys to be printed. If None, print
            all stored keys, default to None.
        """
        if "update/env_step" not in self.logger_keys:
            self.store(tab="update", env_step=step)
        # save .txt file to the output logger
        if self.output_file is not None:
            if self.first_row:
                keys = ["Steps"] + list(self.logger_keys)
                self.output_file.write("\t".join(keys) + "\n")
            vals = [step] + self.get_mean_list(self.logger_keys)
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
            self.first_row = False
        if display:
            self.display_tabular(display_keys=display_keys)
        self.reset_data()

    def write_without_reset(self, *args, **kwarg) -> None:
        """Writing data to somewhere without resetting the current stored stats, \
            for tensorboard and wandb logger usage."""

    def save_checkpoint(self, suffix: Optional[Union[int, str]] = None) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param Optional[Union[int, str]] suffix: the suffix to be added to the stored
            checkpoint name, defaults to None.
        """
        if self.checkpoint_fn and self.log_dir:
            fpath = osp.join(self.log_dir, "checkpoint")
            os.makedirs(fpath, exist_ok=True)
            suffix = '%d' % suffix if isinstance(suffix, int) else suffix
            suffix = '_' + suffix if suffix is not None else ""
            fname = 'model' + suffix + '.pt'
            torch.save(self.checkpoint_fn(), osp.join(fpath, fname))

    def save_config(self, config: dict, verbose=True) -> None:
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important config
        vars as a dict. This will serialize the config to JSON, while handling anything
        which can't be serialized in a graceful way (writing as informative a string as
        possible).

        Example use:

        .. code-block:: python

            logger = BaseLogger(**logger_kwargs) logger.save_config(locals())

        :param dict config: the configs to be stored.
        :param bool verbose: whether to print the saved configs, default to True.
        """
        if self.name is not None:
            config['name'] = self.name
        config_json = convert_json(config)
        if verbose:
            print(colorize('Saving config:\n', color='cyan', bold=True))
            output = json.dumps(
                config_json, separators=(',', ':\t'), indent=4, sort_keys=True
            )
            print(output)
        if self.log_dir:
            with open(osp.join(self.log_dir, "config.yaml"), 'w') as out:
                yaml.dump(
                    config, out, default_flow_style=False, indent=4, sort_keys=False
                )

    def restore_data(self) -> None:
        """Return the metadata from existing log. Not implemented for BaseLogger.
        """
        pass

    def get_std(self, key: str) -> float:
        """Get the standard deviation of the queried data in storage.

        :param str key: the key of the queried data.
        :return: the standard deviation.
        """
        return self.log_data[key].std

    def get_mean(self, key: str) -> float:
        """Get the mean of the queried data in storage.

        :param str key: the key of the queried data.
        :return: the mean.
        """
        return self.log_data[key].mean

    def get_mean_list(self, keys: Iterable[str]) -> list:
        """Get the list of queried data in storage.

        :param Iterable[str] keys: the keys of the queried data.
        :return: the list of mean values.
        """
        return [self.get_mean(key) for key in keys]

    def get_mean_dict(self, keys: Iterable[str]) -> dict:
        """Get the dict of queried data in storage.

        :param Iterable[str] keys: the keys of the queried data.

        :return: the dict of mean values.
        """
        return {key: self.get_mean(key) for key in keys}

    @property
    def stats_mean(self) -> dict:
        return self.get_mean_dict(self.logger_keys)

    @property
    def logger_keys(self) -> Iterable:
        return self.log_data.keys()

    def display_tabular(self, display_keys: Iterable[str] = None) -> None:
        """Display the keys of interest in a tabular format.

        :param Iterable[str] display_keys: the keys to be displayed, if None, display
            all data. defaults to None.
        """
        if not display_keys:
            display_keys = sorted(self.logger_keys)
        key_lens = [len(key) for key in self.logger_keys]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d' % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        for key in display_keys:
            val = self.log_data[key].mean
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
        print("-" * n_slashes, flush=True)

    def print(self, msg: str, color='green') -> None:
        """Print a colorized message to stdout.

        :param str msg: the string message to be printed
        :param str color: the colors for printing, the choices are ```gray, red, green,
            yellow, blue, magenta, cyan, white, crimson```. Default to "green".
        """
        print(colorize(msg, color, bold=True))


class DummyLogger(BaseLogger):
    """A logger that inherent from the BaseLogger but does nothing. \
         Used as the placeholder in trainer."""

    def __init__(self, *args, **kwarg) -> None:
        pass

    def setup_checkpoint_fn(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def store(self, *args, **kwarg) -> None:
        """The DummyLogger stores nothing"""

    def reset_data(self, *args, **kwarg) -> None:
        """The DummyLogger resets nothing"""

    def write(self, *args, **kwarg) -> None:
        """The DummyLogger writes nothing."""

    def write_without_reset(self, *args, **kwarg) -> None:
        """The DummyLogger writes nothing"""

    def save_checkpoint(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def save_config(self, *args, **kwarg) -> None:
        """The DummyLogger saves nothing"""

    def restore_data(self, *args, **kwarg) -> None:
        """The DummyLogger restores nothing"""

    def get_mean(self, *args, **kwarg) -> float:
        """The DummyLogger returns 0"""
        return 0

    def get_std(self, *args, **kwarg) -> float:
        """The DummyLogger returns 0"""
        return 0

    def get_mean_list(self, *args, **kwarg) -> None:
        """The DummyLogger returns nothing"""

    def get_mean_dict(self, *args, **kwarg) -> None:
        """The DummyLogger returns nothing"""

    @property
    def stats_mean(self) -> None:
        """The DummyLogger returns nothing"""

    @property
    def logger_keys(self) -> None:
        """The DummyLogger returns nothing"""
