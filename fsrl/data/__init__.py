"""Data package."""
# isort:skip_file

from fsrl.data.fast_collector import FastCollector
from fsrl.data.basic_collector import BasicCollector
from fsrl.data.traj_buf import TrajectoryBuffer

__all__ = ["FastCollector", "BasicCollector", "TrajectoryBuffer"]
