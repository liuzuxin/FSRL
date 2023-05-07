import json
from typing import Union

import numpy as np

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON.
    Reference: https://github.com/openai/spinningup
    """
    try:
        # the object is json serializable, just return it
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v)
                for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}
        return str(obj)


class RunningAverage(object):
    """Computes running mean and standard deviation.
    Reference: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    """

    def __init__(self, mean=0., vars=0., count=0) -> None:
        self.mean, self.vars = mean, vars
        self.count = count

    def reset(self):
        self.count = 0

    def add(self, x: Union[int, float]) -> None:
        """Add a number to the running average, update mean/std/count."""
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.vars = 0.
        else:
            prev_mean = self.mean
            self.mean += (x - self.mean) / self.count
            self.vars += (x - prev_mean) * (x - self.mean)

    def __add__(self, other):
        assert isinstance(other, RunningAverage)
        sum_ns = self.count + other.count
        prod_ns = self.count * other.count
        delta2 = (other.mean - self.mean)**2.
        return RunningAverage(
            (self.mean * self.count + other.mean * other.count) / sum_ns,
            self.vars + other.vars + delta2 * prod_ns / sum_ns, sum_ns
        )

    @property
    def var(self):
        return self.vars / (self.count) if self.count else 0.0

    @property
    def std(self):
        return np.sqrt(self.var)

    def __repr__(self):
        # return '<RunningAverage(mean={: 2.4f}, std={: 2.4f}, count={: 2f})>'.format(
        #     self.mean, self.std, self.count)
        return '{: .3g}'.format(self.mean)

    def __str__(self):
        return 'mean={: .3g}, std={: .3g}'.format(self.mean, self.std)

    def __call__(self):
        return self.mean


def test():
    from collections import defaultdict
    running_averages = [defaultdict(RunningAverage) for _ in range(2)]
    data = np.arange(10)
    for d in data[:5]:
        running_averages[0]["k"].add(d)
    print(running_averages[0]["k"])
    print(
        "numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data[:5]), np.std(data[:5]))
    )

    for d in data[5:]:
        running_averages[1]["k"].add(d)
    print(running_averages[1]["k"])
    print(
        "numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data[5:]), np.std(data[5:]))
    )

    print("Testing summation")
    print(running_averages[0]["k"] + running_averages[1]["k"])
    print("numpy mean={: 2.4f}, std={: 2.4f}".format(np.mean(data), np.std(data)))


if __name__ == "__main__":
    test()
