import os
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
from tianshou.data import Batch
from tianshou.data.utils.converter import to_hdf5


class TrajectoryBuffer:
    """Buffer for storing trajectories collected during training.

    If use grid filter, it will discard exceeded trajectories based on the density over
    the cost-return and reward-return space. It will only store the trajectory in the
    buffer if its reward return and cost return are  within the user- defined ranges:
    `rmin`, `rmax`, `cmin`, `cmax`.

    :param int max_trajectory: Maximum number of trajectories to store. (default=99999)
    :param bool use_grid_filter: If True, use grid filtering to downsample the data.
        (default=True)
    :param float rmin: The minimum reward return of trajectory that can be stored in the
        buffer
    :param float rmax: The maximum reward return of trajectory that can be stored in the
        buffer
    :param float cmin: The minimum cost return of trajectory that can be stored in the
        buffer
    :param float cmax: The maximum cost return of trajectory that can be stored in the
        buffer
    :param float filter_interval: Only used when use_grid_filter is True. The filter
        interval is the ratio of trajectory numbers to keep in the buffer. (default=2.0)
    """

    def __init__(
        self,
        max_trajectory: int = 99999,
        use_grid_filter: bool = True,
        rmin: float = -np.inf,
        rmax: float = np.inf,
        cmin: float = -np.inf,
        cmax: float = np.inf,
        filter_interval: float = 2
    ):
        self.max_trajectory = max_trajectory
        self.buffer: List[Batch] = []
        self.current_trajectory = Batch()
        self.current_rew, self.current_cost = 0, 0
        self.metrics: List[np.ndarray] = []
        self.rmin = rmin
        self.rmax = rmax
        self.cmin = cmin
        self.cmax = cmax

        self.use_grid_filter = use_grid_filter
        if self.use_grid_filter:
            assert filter_interval > 1, "the filter interval should be greater than 1"
            self.filtering_thres = int(filter_interval * max_trajectory)

    def store(self, data: Batch) -> None:
        """Stores a batch of data in the buffer.

        :param Batch data: Batch of data to store.
        """
        # Concatenate data to the current trajectory
        self.current_trajectory = Batch.cat([self.current_trajectory, data])
        done = data["terminals"].item() or data["timeouts"].item()
        self.current_rew += data["rewards"].item()
        self.current_cost += data["costs"].item()
        if done:
            if self.current_rew > self.rmax or self.current_rew < self.rmin \
                    or self.current_cost > self.cmax or self.current_cost < self.cmin:
                pass
            else:
                if len(self.buffer) < self.max_trajectory:
                    self.buffer.append(self.current_trajectory)
                    self.metrics.append(np.array([self.current_rew, self.current_cost]))
                else:
                    if self.use_grid_filter:
                        self.buffer.append(self.current_trajectory)
                        self.metrics.append(
                            np.array([self.current_rew, self.current_cost])
                        )
                        # apply grid filter when the buffer size reaches the
                        # filtering_thres
                        if len(self.buffer) >= self.filtering_thres:
                            self.apply_grid_filter()
                    else:
                        idx_to_replace = np.random.randint(0, len(self.buffer))
                        self.buffer[idx_to_replace] = self.current_trajectory
                        self.metrics[idx_to_replace] = np.array(
                            [self.current_rew, self.current_cost]
                        )
            self.current_trajectory = Batch()
            self.current_rew, self.current_cost = 0, 0

    def apply_grid_filter(self) -> None:
        """Apply grid filtering to the buffer and metrics data.

        The filter will removing some trajectories with the highest density.

        --- Note: This method modifies the `buffer` and `metrics` arrays in place.
        """
        kept_idxs = self.filter_points(self.metrics, self.max_trajectory)
        # keep the data in the kept idxs and remove others; in-place operation
        indices_set = set(kept_idxs)
        write_index = 0

        for read_index in range(len(self.buffer)):
            if read_index in indices_set:
                if read_index != write_index:
                    self.buffer[write_index] = self.buffer[read_index]
                    self.metrics[write_index] = self.metrics[read_index]
                write_index += 1

        del self.buffer[write_index:]
        del self.metrics[write_index:]

    @staticmethod
    def filter_points(points: list, target_size: int) -> list:
        """Filter a list of 2D points and returns a list of filtered indices.

        The filtering is done by keeping a certain number of points (determined by the
        target_size parameter) while trying to preserve the spatial distribution of the
        original points as much as possible.

        :param points: A list of 2D points represented as a numpy array of shape (N, 2).
        :param target_size: The number of points to keep after filtering.

        :return: A list of indices that represent the filtered points.
        """
        points = np.array(points)
        grid_size = int(np.ceil(np.sqrt(target_size)))
        # create the grid to store the frequency
        grid_range = [(points[:, i].min(), points[:, i].max()) for i in range(2)]
        cell_size = [(r[1] - r[0]) / grid_size for r in grid_range]

        grid = defaultdict(list)
        for i, point in enumerate(points):
            cell = tuple(
                int((point[i] - grid_range[i][0]) // cell_size[i]) for i in range(2)
            )
            grid[cell].append(i)

        kept_idxs = []
        # First, add one point from each non-empty cell
        for pt_idxs in grid.values():
            if len(pt_idxs) > 0:
                idx = pt_idxs.pop()
                kept_idxs.append(idx)

        # If the number of reduced points is less than target_size, add more points
        non_empty_cells = [cell for cell, points in grid.items() if len(points) > 0]
        while len(kept_idxs) < target_size:
            cell = random.choice(non_empty_cells)
            idx = grid[cell].pop()
            kept_idxs.append(idx)
            if len(grid[cell]) == 0:
                non_empty_cells.remove(cell)

        return kept_idxs[:target_size]

    def __len__(self) -> int:
        return sum([len(traj) for traj in self.buffer])

    def sample(self, batch_size: int) -> Batch:
        """Samples a batch of transitions from the buffer.

        :param int batch_size: Number of transitions to sample.

        :return: Batch of sampled transitions.
        """
        num_trajectories = len(self.buffer)
        traj_indices = np.random.randint(0, num_trajectories, size=batch_size)
        sampled_batch = Batch()
        for i in range(batch_size):
            sampled_traj = self.buffer[traj_indices[i]]
            transition_idx = np.random.randint(0, len(sampled_traj))
            sampled_transition = sampled_traj[transition_idx]
            Batch.cat(sampled_batch, sampled_transition)
        return sampled_batch

    def get_all(self) -> Batch:
        """Returns all the transitions stored in the buffer as a single batch.

        :return: All stored transitions as a single batch.
        :rtype: Batch
        """
        return Batch.cat(self.buffer)

    def save(self, log_dir: str, dataset_name: str = "dataset.hdf5") -> None:
        """Saves the entire buffer to disk as an HDF5 file.

        :param log_dir: Directory to save the dataset in.
        :type log_dir: str
        :param dataset_name: Name of the dataset file to save.
        :type dataset_name: str, optional (default="dataset.hdf5")
        """
        print("Saving dataset...")
        if not os.path.exists(log_dir):
            print(f"Creating saving dir {log_dir}")
            os.makedirs(log_dir)
        dataset_path = os.path.join(log_dir, dataset_name)
        all_data = self.get_all()
        with h5py.File(dataset_path, "w") as f:
            to_hdf5(all_data, f, compression='gzip')
        print(f"Finish saving dataset to {dataset_path}!")
