<div align="center">
  <a href="http://fsrl.readthedocs.io"><img width="300px" height="auto" src="https://github.com/liuzuxin/fsrl/raw/main/docs/_static/images/fsrl-logo.png"></a>
</div>

<br/>

<div align="center">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  [![Documentation Status](https://img.shields.io/readthedocs/fsrl?logo=readthedocs)](https://fsrl.readthedocs.io)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
  [![CodeCov](https://codecov.io/github/liuzuxin/fsrl/branch/main/graph/badge.svg?token=BU27LTW9F3)](https://codecov.io/github/liuzuxin/fsrl)
  [![Tests](https://github.com/liuzuxin/fsrl/actions/workflows/test.yml/badge.svg)](https://github.com/liuzuxin/fsrl/actions/workflows/test.yml)
  <!-- [![CodeCov](https://img.shields.io/codecov/c/github/liuzuxin/fsrl/main?logo=codecov)](https://app.codecov.io/gh/liuzuxin/fsrl) -->
  <!-- [![tests](https://img.shields.io/github/actions/workflow/status/liuzuxin/fsrl/test.yml?label=tests&logo=github)](https://github.com/liuzuxin/fsrl/tree/HEAD/tests) -->
  <!-- [![PyPI](https://img.shields.io/pypi/v/fsrl?logo=pypi)](https://pypi.org/project/fsrl) -->
  <!-- [![GitHub Repo Stars](https://img.shields.io/github/stars/liuzuxin/fsrl?color=brightgreen&logo=github)](https://github.com/liuzuxin/fsrl/stargazers)
  [![Downloads](https://static.pepy.tech/personalized-badge/fsrl?period=total&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/fsrl) -->
   <!-- [![License](https://img.shields.io/github/license/liuzuxin/fsrl?label=license)](#license) -->

</div>

<p align="center">
<a href="https://github.com/liuzuxin/fsrl#-key-features">Key Features</a> |
  <a href="https://github.com/liuzuxin/fsrl#-documentation">Documentation</a> |
  <a href="https://github.com/liuzuxin/fsrl#%EF%B8%8F-installation">Installation</a> |
  <a href="https://github.com/liuzuxin/fsrl#--quick-start">Quick Start</a> |
  <a href="https://github.com/liuzuxin/fsrl#contributing">Contributing</a>
</p>

---

The Fast Safe Reinforcement Learning (FSRL) package provides modularized implementations
of Safe RL algorithms based on PyTorch and the [Tianshou](https://tianshou.readthedocs.io/en/master/) framework. Safe RL is a rapidly evolving subfield of RL, focusing on ensuring the safety of learning agents during the training and deployment process. The study of Safe RL is essential because it addresses the critical challenge of preventing unintended or harmful actions while still optimizing an agent's performance in complex environments.

This project offers high-quality and fast implementations of popular Safe RL algorithms, serving as an ideal starting point for those looking to explore and experiment in this field. By providing a comprehensive and accessible toolkit, the FSRL package aims to accelerate research in this crucial area and contribute to the development of safer and more reliable RL-powered systems.

**Please note that this project is still under active development, and major updates might be expected.** Your feedback and contributions are highly appreciated, as they help us improve the FSRL package.

## 🌟 Key Features

FSRL is designed with several key aspects in mind:

- **High-quality implementations**. For instance, the [CPO]((https://arxiv.org/abs/1705.10528)) implementation by SafetyGym fails to satisfy constraints according to their [benchmark results](https://openai.com/research/safety-gym). As a result, many safe RL papers that adopt these implementations may also report failure results. However, we discovered that with appropriate hyper-parameters and our implementation, it can achieve good safety performance in most tasks as well.
- **Fast training speed**. FSRL cares about accelerating experimentation and benchmarking processes, providing fast training times for popular safe RL tasks. For example, most algorithms can solve the [SafetyCarCircle-v0](https://github.com/liuzuxin/Bullet-Safety-Gym/tree/master#tasks) task in 10 minutes with 4 cpus. The [CVPO](https://arxiv.org/abs/2201.11927) algorithm implementation can also achieve 5x faster training than the original implementation.
- **Well-tuned hyper-parameters**. We carefully studied the effects of key hyperparameters for these algorithms and plan to provide a practical guide for tuning them. We believe both implementations and hyper-parameters play a crucial role in learning a successful safe RL agent.
- **Modular design and easy usability**. FSRL is built upon the elegant RL framework [Tianshou](https://tianshou.readthedocs.io/en/master/). We provide an agent wrapper, refactored loggers for both Tensorboard and Wandb, and [pyrallis](https://github.com/eladrich/pyrallis) configuration support to further facilitate usage. Our algorithms also support multiple constraints and standard RL tasks (like Mujoco).

The implemented safe RL algorithms include:

| Algorithm        | Type           | Description           |
|:-------------------:|:-----------------:|:------------------------:|
| CPO                   | on-policy           | [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) |
| FOCOPS            | on-policy           | [First Order Constrained Optimization in Policy Space](https://arxiv.org/abs/2002.06506) |
| PPOLagrangian  | on-policy           | [PPO](https://arxiv.org/abs/1707.06347) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| TRPOLagrangian | on-policy           | [TRPO](https://arxiv.org/abs/1502.05477) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| DDPGLagrangian | off-on-policy <sup>**(1)**</sup> | [DDPG](https://arxiv.org/abs/1509.02971) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) |
| SACLagrangian   | off-on-policy <sup>**(1)**</sup> | [SAC](https://arxiv.org/abs/1801.01290) with [PID Lagrangian](https://arxiv.org/abs/2007.03964) | 
| CVPO                | off-policy           | [Constrained Variational Policy Optimization](https://arxiv.org/abs/2201.11927) |

<sup>(1): **Off-on-policy** means that the base learning algorithm is off-policy, but the Lagrange multiplier is updated in an on-policy fashion. Our previous [finding](https://arxiv.org/abs/2201.11927) suggested that using off-policy style Lagrange update may result in poor performance</sup><br/>

The implemented algorithms are well-tuned for many tasks in the following safe RL
environments, which cover the majority of tasks in recent safe RL papers:

- [BulletSafetyGym](https://github.com/liuzuxin/Bullet-Safety-Gym), FSRL will install
  this environment by default as the testing ground.
- [SafetyGymnasium](https://github.com/OmniSafeAI/safety-gymnasium), note that you need
  to install it from the source because our current version adopts the `gymnasium` API.

Note that the latest versions of FSRL and the above environments use the `gymnasium >= 0.26.3` API. But if you want to use the old `gym` API such as the `safety_gym`, you can simply change the example scripts from `import gymnasium as gym` to `import gym`.

## 🔍 Documentation

The tutorials and API documentation are hosted on [fsrl.readthedocs.io](https://fsrl.readthedocs.io/).

The majority of the API design in FSRL follows Tianshou, and we aim to reuse their modules as much as possible. For example, the [Env](https://tianshou.readthedocs.io/en/master/api/tianshou.env.html),
[Batch](https://tianshou.readthedocs.io/en/master/tutorials/batch.html),
[Buffer](https://tianshou.readthedocs.io/en/master/api/tianshou.data.html#buffer),
and (most) [Net](https://tianshou.readthedocs.io/en/master/api/tianshou.utils.html#pre-defined-networks) modules are used directly in our repo. This means that you can refer to their comprehensive documentation to gain a good understanding of the code structure. We highly recommend you read the following Tianshou tutorials:

- [Get Started with Jupyter Notebook](https://tianshou.readthedocs.io/en/master/tutorials/get_started.html). You can get a quick overview of different modules through this tutorial.
- [Basic concepts in Tianshou](https://tianshou.readthedocs.io/en/master/tutorials/concepts.html). Note that the basic concepts in FSRL are almost the same as Tianshou.
- [Understanding Batch](https://tianshou.readthedocs.io/en/master/tutorials/batch.html). Note that the Batch data structure is extensively used in this repo.

We observe that for most existing safe RL environments, a few layers of neural networks can solve them quite effectively. Therefore, we provide an 'Agent' class with default MLP networks to facilitate the usage.
You can refer to the [tutorial](#agent) for more details.

Example training and evaluation scripts for both default MLP agent and customized networks are available at the [examples](https://github.com/liuzuxin/fsrl/blob/main/examples) folder.


## 🛠️ Installation

FSRL requires Python >= 3.8. You can install it from source by:
```shell
git clone https://github.com/liuzuxin/fsrl.git
cd fsrl
pip install -e .
```
<!-- It is currently hosted on [PyPI](https://pypi.org/project/fsrl/). 
You can simply install FSRL with the following command:
```shell
pip install fsrl
``` -->

<!-- You can also install with the newest version through GitHub: -->
You can also directly install it with pip through GitHub:
```shell
pip install git+https://github.com/liuzuxin/fsrl.git@main --upgrade
```

You can check whether the installation is successful by:
```python
import fsrl
print(fsrl.__version__)
```

## 🚀  Quick Start

### Training with default MLP agent
<a name="agent"></a>

This is an example of training a PPO-Lagrangian agent with a Tensorboard logger and default parameters.

First, import relevant packages:
```python
import bullet_safety_gym
import gymnasium as gym
from tianshou.env import DummyVectorEnv
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
```

Then initialize the environment, logger, and agent:
```python
task = "SafetyCarCircle-v0"
# init logger
logger = TensorboardLogger("logs", log_txt=True, name=task)
# init the PPO Lag agent with default parameters
agent = PPOLagAgent(gym.make(task), logger)
# init the envs
training_num, testing_num = 10, 1
train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])
```

Finally, start training:
```python
agent.learn(train_envs, test_envs, epoch=100)
```
You can check the experiment results in the `logs/SafetyCarCircle-v0` folder.

### Training with the example scripts

We provide easy-to-use example training script for all the agents in the `examples/mlp` folder. Each training script is by default use the [Wandb](https://wandb.ai/site) logger and [Pyrallis](https://github.com/eladrich/pyrallis) configuration system. The default hyper-parameters are located the `fsrl/config` folder. 
You have three alternatives to run the experiment with your customized hyper-parameters:

#### M1. Directly override the parameters via the command line:
```shell
python examples/mlp/train_ppol_agent.py --arg value --arg2 value2 ...
```
where `--arg` specify the parameter you want to override. For example, `--task SafetyAntRun-v0`. Note that if you specify `--use_default_cfg 1`, the script will automatically use the task's default parameters for training. We plan to release more default configs in the future.

#### M2. Use pre-defined `yaml` or `json` or `toml` configs.
For example, you want to use a different learning-rate and training epochs from our default ones, create a `my_cfg.yaml`:
```yaml
task: "SafetyDroneCircle-v0"
epoch: 500
lr: 0.001
```
Then you can starting training with above parameters by: 
```shell
python examples/mlp/train_ppol_agent.py --config my_cfg.yaml
```
where `--config` specify the path of the configuration parameters.

#### M3. Inherent the config dataclass in the `fsrl/config` folder.
For example, you can inherent the `PPOLagAgent` config by:
```python
from dataclasses import dataclass
from fsrl.config.ppol_cfg import TrainCfg

@dataclass
class MyCfg(TrainCfg):
    task: str = "SafetyDroneCircle-v0"
    epoch: int = 500
    lr: float = 0.001

@pyrallis.wrap()
def train(args: MyCfg):
    ...
```
Then, you can start training with your own default configs:
```shell
python examples/mlp/train_ppol_agent.py
```

Note that our example scripts support the `auto_name` feature, meaning that it can automatically compare your specified hyper-parameters with our default ones, and create the experiment name based on the difference. The default training statistics are saved in the `logs` directory.

### Training with cutomized networks
While the pre-defined MLP agent is sufficient for solving many existing safe RL benchmarks, for more complex tasks, it may be necessary to customize the value and policy networks. Our modular design supports Tianshou's style training scripts. Example training scripts can be found in the `examples/customized` folder. For more details on building networks, please refer to Tianshou's [tutorial]((https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network)), as our algorithms are mostly compatible with their networks.

### Evaluate trained models
To evaluate a trained model, for example, a pre-trained PPOLag model in the `logs/exp_name` folder, run:
```shell
python examples/mlp/eval_ppol_agent.py --path logs/exp_name --eval_episodes 20
```
It will load the saved `config.yaml` from `logs/exp_name/config.yaml` and pre-trained model from `logs/exp_name/checkpoint/model.pt`, run 20 episodes and print the average reward and cost. If the `best` model is saved during training, you can evaluate it by setting `--best 1`.



## Related Projects

FSRL is heavily inspired by the [Tianshou](https://tianshou.readthedocs.io/en/master/) project. In addition, there are several other remarkable safe RL-related projects:

- [Safety-Gymnasium](https://github.com/OmniSafeAI/safety-gymnasium), a well-maintained and customizable safe RL environments based on Mujoco.
- [Bullet-Safety-Gym](https://github.com/liuzuxin/Bullet-Safety-Gym), a tuned and fast safe RL environments based on PyBullet.
- [Safe-Multi-Agent-Mujoco](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco), a multi-agent safe RL environments based on Mujoco.
- [Safe-Control-Gym](https://github.com/utiasDSL/safe-control-gym), a learning-based control and RL library with PyBullet.
- [OmniSafe](https://github.com/OmniSafeAI/omnisafe), a well-maintained infrastructural framework for safe RL algorithms.
- [SafePO](https://github.com/OmniSafeAI/Safe-Policy-Optimization), another benchmark repository for safe RL algorithms.

## Contributing

The main maintainers of this project are: [Zuxin Liu](https://zuxin.me/), [Zijian Guo](https://github.com/Ja4822).

If you have any suggestions or find any bugs, please feel free to submit an issue or a pull request. We welcome contributions from the community! 

