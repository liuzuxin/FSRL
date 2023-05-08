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
    # CVPO arguments
    estep_iter_num: int = 1
    estep_kl: float = 0.02
    estep_dual_max: float = 20
    estep_dual_lr: float = 0.02
    sample_act_num: int = 16
    mstep_iter_num: int = 1
    mstep_kl_mu: float = 0.005
    mstep_kl_std: float = 0.0005
    mstep_dual_max: float = 0.5
    mstep_dual_lr: float = 0.1
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    gamma: float = 0.97
    n_step: int = 2
    tau: float = 0.05
    hidden_sizes: Tuple[int, ...] = (128, 128)
    double_critic: bool = False
    conditioned_sigma: bool = True
    unbounded: bool = False
    last_layer_scale: bool = False
    # collecting params
    epoch: int = 200
    episode_per_collect: int = 10
    step_per_epoch: int = 10000
    update_per_step: float = 0.2
    buffer_size: int = 200000
    worker: str = "ShmemVectorEnv"
    training_num: int = 20
    testing_num: int = 2
    # general train params
    batch_size: int = 256
    reward_threshold: float = 10000  # for early stop purpose
    save_interval: int = 4
    deterministic_eval: bool = True
    action_scaling: bool = True
    action_bound_method: str = "clip"
    resume: bool = False  # TODO
    save_ckpt: bool = True  # set this to True to save the policy model
    verbose: bool = False
    render: bool = False
    # logger params
    logdir: str = "logs"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "cvpo"
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
    # Drone-Run
    # estep_kl: float = 0.001
    # mstep_kl_mu: float = 0.0002
    # mstep_kl_std: float = 0.0001


# safety gymnasium task default configs


@dataclass
class MujocoBaseCfg(TrainCfg):
    task: str = "SafetyPointCircle1-v0"
    epoch: int = 250
    cost_limit: float = 25
    unbounded: bool = True
    gamma: float = 0.995
    n_step: int = 3
    # collecting params
    step_per_epoch: int = 20000
    buffer_size: int = 200000


@dataclass
class Mujoco2MCfg(MujocoBaseCfg):
    epoch: int = 100


@dataclass
class Mujoco5MCfg(MujocoBaseCfg):
    epoch: int = 250
    unbounded: bool = False
    gamma: float = 0.98
    n_step: int = 3
    buffer_size: int = 40000


@dataclass
class Mujoco20MCfg(MujocoBaseCfg):
    epoch: int = 1000
    sample_act_num: int = 64


@dataclass
class Mujoco10MCfg(MujocoBaseCfg):
    epoch: int = 500
    unbounded: bool = False
    gamma: float = 0.98
    sample_act_num: int = 32
