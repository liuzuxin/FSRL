.. FSRL documentation master file.


Welcome to the FSRL documentation!
===================================

The `FSRL <https://github.com/liuzuxin/fsrl>`_ (Fast Safe Reinforcement Learning) package contains modularized implementations
of safe RL algorithms based on PyTorch and the Tianshou framework :cite:`tianshou`. The
implemented safe RL algorithms include:

* :class:`~fsrl.policy.PPOLagrangian`: `PPO <https://arxiv.org/abs/1707.06347>`_ with `PID Lagrangian <https://arxiv.org/abs/2007.03964>`_, on-policy algorithm, :cite:`stooke2020responsive`
* :class:`~fsrl.policy.TRPOLagrangian`: `TRPO <https://arxiv.org/abs/1502.05477>`_ with `PID Lagrangian <https://arxiv.org/abs/2007.03964>`_, on-policy algorithm, :cite:`stooke2020responsive`
* :class:`~fsrl.policy.SACLagrangian`: `SAC <https://arxiv.org/abs/1801.01290>`_ with `PID Lagrangian <https://arxiv.org/abs/2007.03964>`_, off-policy algorithm with on-policy Lagrangian, :cite:`stooke2020responsive`
* :class:`~fsrl.policy.DDPGLagrangian`: `DDPG <https://arxiv.org/abs/1509.02971>`_ with `PID Lagrangian <https://arxiv.org/abs/2007.03964>`_, off-policy algorithm with on-policy Lagrangian, :cite:`stooke2020responsive`
* :class:`~fsrl.policy.CVPO` `Constrained Varitional Policy Optimization <https://arxiv.org/abs/2201.11927>`_, off-policy algorithm, :cite:`liu2022constrained`
* :class:`~fsrl.policy.CPO` `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`_, on-policy algorithm, :cite:`achiam2017constrained`
* :class:`~fsrl.policy.FOCOPS` `First Order Constrained Optimization in Policy Space <https://arxiv.org/abs/2002.06506>`_, on-policy algorithm, :cite:`zhang2020first`

The implemented algorithms are well-tuned for many tasks in the following safe RL
environments, which cover most tasks in safe RL papers:

- `BulletSafetyGym <https://github.com/liuzuxin/Bullet-Safety-Gym>`_, FSRL will install
  this environment by default as the testing ground.
- `SafetyGymnasium <https://github.com/OmniSafeAI/safety-gymnasium>`_, note that you need
  to install it from the source because we use the `gymnasium` API.

FSRL cares about **implementation** and **hyper-parameters**, as both of them play a
crucial role in successfully training a safe RL agent.

* For instance, the :class:`~fsrl.policy.CPO` method fails to satisfy constraints based
  on the SafetyGym benchmark results and their implementations. As a result, many safe RL
  papers that adopt these implementations may also report failure results. However, we
  discovered that with appropriate hyper-parameters and implementation, it can achieve
  good safety performance in most tasks as well.
* Another example is the off-policy Lagrangian methods:
  :class:`~fsrl.policy.SACLagrangian`, :class:`~fsrl.policy.DDPGLagrangian`. While they
  may fail with off-policy style Lagrangian multiplier updates
  :cite:`liu2022constrained`, they can achieve sample-efficient training and good
  performance with on-policy style Lagrange updates.
* Therefore, we plan to provide a practical guide for tuning the key hyper-parameters of
  safe RL algorithms, which empirically summarize the their effects on the performance.

FSRL cares about the **training speed**, with the aim to accelerate the experimentation
and benchmarking process.

* For example, most algorithms can solve the `SafetyCarRun-v0
  <https://github.com/liuzuxin/Bullet-Safety-Gym/tree/master#tasks>`_ task in 2 minutes
  and the `SafetyCarCircle-v0
  <https://github.com/liuzuxin/Bullet-Safety-Gym/tree/master#tasks>`_ task in 10 minutes with 4 cpus.
  The CVPO algorithm implementation can also achieve 5x faster training than the original repo.
* We also plan to provide a guide regarding how to accelerate your safe RL experiments.


Here are FSRL's other features:

* Elegant framework with modularized implementation, which are mostly the same as
  `Tianshou <https://tianshou.readthedocs.io/en/master/>`_.
* State-of-the-art benchmark performance on popular safe RL tasks.
* Support fast vectorized environment `parallel sampling
  <https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#parallel-sampling>`_
  for all algorithms.
* Support n-step returns estimation
  :meth:`~fsrl.policy.BasePolicy.compute_nstep_returns`; GAE and nstep are very fast
  thanks to numba jit function and vectorized numpy operation.
* Support both `TensorBoard <https://www.tensorflow.org/tensorboard>`_ and `W&B
  <https://wandb.ai/>`_ log tools with customized easy-to-use features.

Checkout the :ref:`get_started` page for more information and start your journey with
FSRL!


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/get_started
   tutorials/benchmark


.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/fsrl.agent
   api/fsrl.policy
   api/fsrl.data
   api/fsrl.trainer
   api/fsrl.utils


.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing
   contributor


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. rubric:: References

.. bibliography:: /refs.bib
