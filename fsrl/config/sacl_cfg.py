from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarCircle-v0"
    cost_limit: float = 10
    device: str = "cpu"
    thread: int = 4  # if use "cpu" to train
    seed: int = 10
    use_default_cfg: bool = False
    # algorithm params
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    auto_alpha: bool = True
    alpha_lr: float = 3e-4
    alpha: float = 0.005
    tau: float = 0.05
    n_step: int = 2
    conditioned_sigma: bool = True
    unbounded: bool = False
    last_layer_scale: bool = False
    # Lagrangian specific arguments
    use_lagrangian: bool = True
    lagrangian_pid: Tuple[float, ...] = (0.05, 0.0005, 0.1)
    rescaling: bool = True
    # Base policy common arguments
    gamma: float = 0.97
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 2
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 100000
    worker: str = "ShmemVectorEnv"
    training_num: int = 10
    testing_num: int = 2
    # general train params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = True
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "sacl"
    suffix: Optional[str] = ""


# bullet-safety-gym task default configs


@dataclass
class Bullet1MCfg(TrainCfg):
    epoch: int = 100


@dataclass
class Bullet5MCfg(TrainCfg):
    epoch: int = 500


@dataclass
class Bullet10MCfg(TrainCfg):
    epoch: int = 1000


# safety gymnasium task default configs


@dataclass
class MujocoBaseCfg(TrainCfg):
    task: str = "SafetyPointCircle1-v0"
    epoch: int = 250
    cost_limit: float = 25
    gamma: float = 0.99
    n_step: int = 3
    # collecting params
    step_per_epoch: int = 20000
    episode_per_collect = 5
    buffer_size: int = 800000


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500
