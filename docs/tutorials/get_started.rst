.. _get_started:

Get Started
============

Installation
------------

.. FSRL is currently hosted on `PyPI <https://pypi.org/project/fsrl/>`_. It requires Python >= 3.8.

.. You can simply install FSRL from PyPI with the following command:

.. .. code-block:: bash

..     $ pip install fsrl

FSRL requires Python >= 3.8. You can install it from source by:

.. code-block:: bash

    $ git clone https://github.com/liuzuxin/fsrl.git
    $ cd fsrl
    $ pip install -e .

You can also install with the newest version through GitHub:

.. code-block:: bash

    $ pip install git+https://github.com/liuzuxin/fsrl.git@main --upgrade

After installation, open your python console and type
::

    import fsrl
    print(fsrl.__version__)

If no error occurs, you have successfully installed FSRL.

Overview
--------

`Tianshou <https://tianshou.readthedocs.io/en/master/>`_ is an excellent RL framework with a clean API design, fast training speed, and good performance. Therefore, the majority of the API design in FSRL follows Tianshou, and we aim to reuse their modules as much as possible.

For example, the `Env <https://tianshou.readthedocs.io/en/master/api/tianshou.env.html>`_,
`Batch <https://tianshou.readthedocs.io/en/master/tutorials/batch.html>`_,
`Buffer <https://tianshou.readthedocs.io/en/master/api/tianshou.data.html#buffer>`_,
and (most) `Net <https://tianshou.readthedocs.io/en/master/api/tianshou.utils.html#pre-defined-networks>`_ modules
are used directly in our repo. This means that you can refer to their comprehensive documentation to gain a
good understanding of the code structure.

However, we have made some major changes to accommodate Tianshou with safe RL tasks, including modifications to :class:`~fsrl.policy.BasePolicy`,
:class:`~fsrl.trainer.BaseTrainer`, :class:`~fsrl.data.FastCollector`, and :class:`~fsrl.utils.BaseLogger`.
We have also added an additional MLP :class:`~fsrl.agent.BaseAgent` class to further facilitate usage,
since we observed that for most existing safe RL environments, a few layers of neural networks can solve them quite effectively.
Nevertheless, we also provide examples of how to customize your own policy network to work together with our implemented policy.

If you want to gain a deep understanding of the code structure and build your own algorithms, we highly recommend you read the following Tianshou tutorials:

* `Get Started with Jupyter Notebook <https://tianshou.readthedocs.io/en/master/tutorials/get_started.html>`_. You can get a quick overview of different modules through this tutorial.
* `Basic concepts in Tianshou <https://tianshou.readthedocs.io/en/master/tutorials/concepts.html>`_. Note that the basic concepts in FSRL are almost the same as Tianshou.
* `Understanding Batch <https://tianshou.readthedocs.io/en/master/tutorials/batch.html>`_. Note that the Batch data structure is extensively used in this repo.

If you only want to test or apply our implemented safe RL agents to your tasks, you can simply follow the instructions here.

.. _agent:

Training with default MLP agent
-------------------------------

This is an example of training a PPO-Lagrangian agent with a Tensorboard logger and default parameters.

First, import relevant packages:

.. code-block:: python

    import bullet_safety_gym
    import gymnasium as gym
    from tianshou.env import DummyVectorEnv
    from fsrl.agent import PPOLagAgent
    from fsrl.utils import TensorboardLogger

Then initialize the environment, logger, and agent:

.. code-block:: python

    task = "SafetyCarCircle-v0"
    # init logger
    logger = TensorboardLogger("logs", log_txt=True, name=task)
    # init the PPO Lag agent with default parameters
    agent = PPOLagAgent(gym.make(task), logger)
    # init the envs
    training_num, testing_num = 10, 1
    train_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
    test_envs = DummyVectorEnv([lambda: gym.make(task) for _ in range(testing_num)])

Finally, start training:

.. code-block:: python

    agent.learn(train_envs, test_envs, epoch=100)

You can check the experiment results in the ``logs/SafetyCarCircle-v0`` folder.

Training with the example scripts
---------------------------------

We provide easy-to-use example training script for all the agents in the ``examples/mlp`` folder. Each training script is by default use the Wandb logger and `Pyrallis <https://github.com/eladrich/pyrallis>`_ configuration system. The default hyper-parameters are located the ``fsrl/config`` folder.
You have three alternatives to run the experiment with your customized hyper-parameters:

M1. Directly override the parameters via the command line:

.. code-block:: shell

    python examples/mlp/train_ppol_agent.py --arg value --arg2 value2 ...

where ``--arg`` specify the parameter you want to override. For example, ``--task SafetyAntRun-v0``. Note that if you specify ``--use_default_cfg 1``, the script will automatically use the task's default parameters for training. We plan to release more default configs in the future.

M2. Use pre-defined ``yaml`` or ``json``` or ``toml`` configs. For example, you want to use a different learning-rate and training epochs from our default ones, create a ``my_cfg.yaml``:

.. code-block:: yaml

    task: "SafetyDroneCircle-v0"
    epoch: 500
    lr: 0.001

Then you can starting training with above parameters by:

.. code-block:: shell

    python examples/mlp/train_ppol_agent.py --config my_cfg.yaml

where ``--config`` specify the path of the configuration parameters.

M3. Inherent the config dataclass in the ``fsrl/config`` folder. For example, you can inherent the ``PPOLagAgent`` config by:

.. code-block:: python

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

Then, you can start training with your own default configs:

.. code-block:: shell

    python examples/mlp/train_ppol_agent.py

Note that our example scripts support the :func:`auto_name <fsrl.utils.exp_util.auto_name>` feature, meaning that it can automatically compare your specified hyper-parameters with our default ones, and create the experiment name based on the difference. The default training statistics are saved in the ``logs`` directory.

Training with customized networks
---------------------------------

While the pre-defined MLP agent is sufficient for solving many existing safe RL benchmarks, for more complex tasks, it may be necessary to customize the value and policy networks. Our modular design supports Tianshou's style training scripts. Example training scripts can be found in the ``examples/customized`` folder. For more details on building networks, please refer to Tianshou's `tutorial <https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network>`_, as our algorithms are mostly compatible with their networks.

Evaluate trained models
-----------------------

To evaluate a trained model, for example, a pre-trained PPOLag model in the ``logs/exp_name`` folder, run:

.. code-block:: shell

    python examples/mlp/eval_ppol_agent.py --path logs/exp_name --eval_episodes 20

It will load the saved ``config.yaml`` from ``logs/exp_name/config.yaml`` and pre-trained model from ``logs/exp_name/checkpoint/model.pt``, run 20 episodes and print the average reward and cost. If the ``best`` model is saved during training, you can evaluate it by setting ``--best 1``.



Logger features
---------------

- Save training configurations to a `.yaml` file and training statistics to a `.txt` file.
- Pass in a `checkpoint_fn` hook to save models.
- Support both Wandb and TensorBoard.
- The :meth:`store <fsrl.utils.BaseLogger.store>` can be called many times during one epoch and the data will be stored in a temporary buffer.
- The :meth:`write <fsrl.utils.BaseLogger.write>` function logs metrics based on the ``average`` of previously stored data.
- Print out stored data in tabular format for specific key metrics.


