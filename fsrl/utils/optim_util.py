import numpy as np


def projection(x):
    return np.maximum(0, x)


class LagrangianOptimizer(object):
    """
    Lagrangian multiplier optimizer based on the PID controller,
    according to https://proceedings.mlr.press/v119/stooke20a.html.

    :param List pid: the coefficients of the PID controller, kp, ki, kd.

    .. note::

        If kp and kd are 0, it reduced to a standard SGD-based Lagrangian optimizer.
    """

    def __init__(self, pid: tuple = (0.05, 0.0005, 0.1)) -> None:
        super().__init__()
        assert len(pid) == 3, " the pid param should be a list with 3 numbers"
        self.pid = tuple(pid)
        self.error_old = 0.
        self.error_integral = 0.
        self.lagrangian = 0.

    def step(self, value: float, threshold: float) -> None:
        """Optimize the multiplier by one step

        :param float value: the current value estimation
        :param float threshold: the threshold of the value
        """
        error_new = np.mean(value - threshold)  # [batch]
        error_diff = projection(error_new - self.error_old)
        self.error_integral = projection(self.error_integral + error_new)
        self.error_old = error_new
        self.lagrangian = projection(
            self.pid[0] * error_new + self.pid[1] * self.error_integral +
            self.pid[2] * error_diff
        )

    def get_lag(self) -> float:
        """Get the lagrangian multiplier."""
        return self.lagrangian

    def state_dict(self) -> dict:
        """Get the parameters of this lagrangian optimizer"""
        params = {
            "pid": self.pid,
            "error_old": self.error_old,
            "error_integral": self.error_integral,
            "lagrangian": self.lagrangian
        }
        return params

    def load_state_dict(self, params: dict) -> None:
        """Load the parameters to continue training"""
        self.pid = params["pid"]
        self.error_old = params["error_old"]
        self.error_integral = params["error_integral"]
        self.lagrangian = params["lagrangian"]
