from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.utils import MultipleLRSchedulers, RunningMeanStd
from torch import nn

from fsrl.utils import BaseLogger
from fsrl.utils.net.common import ActorCritic


class BasePolicy(ABC, nn.Module):
    """The base class for safe RL policy.

    The base class follows a similar structure as `Tianshou
    <https://github.com/thu-ml/tianshou>`_. All of the policy classes must inherit
    :class:`~fsrl.policy.BasePolicy`.

    A policy class typically has the following parts:

    * :meth:`~srl.policy.BasePolicy.__init__`: initialize the policy, including coping
        the target network and so on;
    * :meth:`~srl.policy.BasePolicy.forward`: compute action with given observation;
    * :meth:`~srl.policy.BasePolicy.process_fn`: pre-process data from the replay buffer
        (this function can interact with replay buffer);
    * :meth:`~srl.policy.BasePolicy.learn`: update policy with a given batch of data.
    * :meth:`~srl.policy.BasePolicy.post_process_fn`: update the replay buffer from the
        learning process (e.g., prioritized replay buffer needs to update the weight);
    * :meth:`~srl.policy.BasePolicy.update`: the main interface for training, i.e.,
        `process_fn -> learn -> post_process_fn`.

    Most of the policy needs a neural network to predict the action and an optimizer to
    optimize the policy. The rules of self-defined networks are:

    1. Input: observation "obs" (may be a ``numpy.ndarray``, a ``torch.Tensor``, a \
    dict or any others), hidden state "state" (for RNN usage), and other information \
    "info" provided by the environment. 2. Output: some "logits", the next hidden state
    "state", and the intermediate result during policy forwarding procedure "policy". The
    "logits" could be a tuple instead of a ``torch.Tensor``. It depends on how the policy
    process the network output. For example, in PPO, the return of the network might be
    ``(mu, sigma), state`` for Gaussian policy. The "policy" can be a Batch of
    torch.Tensor or other things, which will be stored in the replay buffer, and can be
    accessed in the policy update process (e.g. in "policy.learn()", the "batch.policy"
    is what you need).

    Since :class:`~fsrl.policy.BasePolicy` inherits ``torch.nn.Module``, you can use
    :class:`~fsrl.policy.BasePolicy` almost the same as ``torch.nn.Module``, for
    instance, loading and saving the model: ::

        torch.save(policy.state_dict(), "policy.pth")
        policy.load_state_dict(torch.load("policy.pth"))

    :param torch.nn.Module actor: the actor network.
    :param Union[nn.Module, List[nn.Module]] critics: the critic network(s). (s -> V(s))
    :param dist_fn: distribution class for stochastic policy to sample the action.
        Default to None :type dist_fn: Type[torch.distributions.Distribution]
    :param BaseLogger logger: the logger instance for logging training information. \
        Default to DummyLogger.
    :param float gamma: the discounting factor for cost and reward, should be in [0, 1].
        Default to 0.99.
    :param int max_batchsize: the maximum size of the batch when computing GAE, depends
        on the size of available memory and the memory cost of the model; should be as
        large as possible within the memory constraint. Default to 99999.
    :param bool reward_normalization: normalize estimated values to have std close to 1,
        also normalize the advantage to Normal(0, 1). Default to False.
    :param deterministic_eval: whether to use deterministic action instead of stochastic
        action sampled by the policy. Default to True.
    :param action_scaling: whether to map actions from range [-1, 1] to range \
        [action_spaces.low, action_spaces.high]. Default to True.
    :param action_bound_method: method to bound action to range [-1, 1]. Default to
        "clip".
    :param observation_space: environment's observation space. Default to None.
    :param action_space: environment's action space. Default to None.
    :param lr_scheduler: learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None.
    """

    def __init__(
        self,
        actor: nn.Module,
        critics: Union[nn.Module, List[nn.Module]],
        dist_fn: Optional[Type[torch.distributions.Distribution]] = None,
        logger: BaseLogger = BaseLogger(),
        gamma: float = 0.99,
        max_batchsize: Optional[int] = 99999,
        reward_normalization: bool = False,
        deterministic_eval: bool = True,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        lr_scheduler: Optional[Union[torch.optim.lr_scheduler.LambdaLR,
                                     MultipleLRSchedulers]] = None
    ) -> None:
        super().__init__()
        self.actor = actor
        if isinstance(critics, nn.Module):
            self.critics = nn.ModuleList([critics])
        elif isinstance(critics, List):
            self.critics = nn.ModuleList(critics)
        else:
            raise TypeError("critics should not be %s" % (type(critics)))
        self.critics_num = len(self.critics)
        self.dist_fn = dist_fn
        self.logger = logger
        self.ret_rms = [RunningMeanStd() for _ in range(self.critics_num)]
        assert 0.0 <= gamma <= 1.0, "discount factor should be in [0, 1]."
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        self._max_batchsize = max_batchsize
        self._actor_critic = ActorCritic(self.actor, self.critics)

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_type = ""
        if isinstance(action_space, (Discrete, MultiDiscrete, MultiBinary)):
            self.action_type = "discrete"
        elif isinstance(action_space, Box):
            self.action_type = "continuous"
        else:
            print("Warning! The action sapce type is unclear, regard it as continuous.")
            self.action_type = "continuous"
            print(self.action_space)
        self.updating = False
        self.action_scaling = action_scaling
        # can be one of ("clip", "tanh", ""), empty string means no bounding
        assert action_bound_method in ("", "clip", "tanh")
        self.action_bound_method = action_bound_method
        self.lr_scheduler = lr_scheduler
        self.gradient_steps = 0
        self._compile()

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which MUST have the following keys:

            * ``act`` an numpy.ndarray or a torch.Tensor, the action over \
                given batch data.
            * ``state`` a dict, an numpy.ndarray or a torch.Tensor, the \
                internal state of the policy, ``None`` as default.

        Other keys are user-defined based on the algorithm. For example, ::

            # for stochastic policy return Batch(logits=..., act=..., state=None,
            dist=...)

            where, * ``logits`` the network's raw output. * ``dist`` the action
            distribution. * ``state`` the hidden state.

        The keyword ``policy`` is reserved and the corresponding data will be stored into
        the replay buffer. For instance, ::

            # some code return Batch(..., policy=Batch(log_prob=dist.log_prob(act))) #
            and in the sampled data batch, you can directly use # batch.policy.log_prob
            to get your data.

        .. note::

            In continuous action space, you should do another step "map_action" to get
            the real action: ::

                act = policy(batch).act  # doesn't map to the target action range act =
                policy.map_action(act, batch)
        """
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def pre_update_fn(self, **kwarg: Any) -> Any:
        """Pre-process the policy or data before updating the policy.

        This function is called after each data collection in :meth:`trainer` and could
        be used to update the Lagrangian multiplier.
        """
        pass

    def post_update_fn(self, **kwarg: Any) -> Any:
        """Post-process the policy or data after updating the policy.

        This function is in :meth:`trainer` and could be used to sync the weight or old
        variables.
        """
        pass

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Modify the action from policy.forward with exploration noise.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.
        :param batch: the input batch for policy.forward, kept for advanced usage.

        :return: action in the same form of input "act" but with added exploration noise.
        """
        return act

    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer and
        thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it to
        [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def map_action_inverse(
        self, act: Union[Batch, List, np.ndarray]
    ) -> Union[Batch, List, np.ndarray]:
        """Inverse operation to :meth:`~fsrl.policy.BasePolicy.map_action`.

        This function is called in :meth:`~fsrl.data.FastCollector.collect` for random
        initial steps. It scales [action_space.low, action_space.high] to the value
        ranges of policy.forward.

        :param act: a data batch, list or numpy.ndarray which is the action taken by
            gym.spaces.Box.sample().

        :return: action remapped.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            act = to_numpy(act)
            if isinstance(act, np.ndarray):
                if self.action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    scale = high - low
                    eps = np.finfo(np.float32).eps.item()
                    scale[scale < eps] += eps
                    act = (act - low) * 2.0 / scale - 1.0
                if self.action_bound_method == "tanh":
                    act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0  # type: ignore
        return act

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out `here
        <https://tianshou.readthedocs.io/en/master/tutorials/concepts.html#process-fn>`_
        for more information.
        """
        return batch

    @abstractmethod
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        """Update policy with a given batch of data.

        :return: A dict, including the data needed to be logged (e.g., loss).

        .. note::

            In order to distinguish the collecting state, updating state and testing
            state, you can check the policy state by ``self.training`` and
            ``self.updating``. Please refer to `this
            <https://tianshou.readthedocs.io/en/master/tutorials/concepts.html#policy-state>`_
            for more detailed explanation.

        .. warning::

            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob, please be
            careful about the shape: Categorical distribution gives "[batch_size]" shape
            while Normal distribution gives "[batch_size, 1]" shape. The
            auto-broadcasting of numerical operation with torch tensors will amplify this
            error.
        """
        pass

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized experience replay.
        Used in :meth:`update`.
        """
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indices, batch.weight)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        """Update the policy network and replay buffer.

        It includes 3 function steps: process_fn, learn, and post_process_fn. In
        addition, this function will change the value of ``self.updating``: it will be
        False before this function and will be True when executing :meth:`update`.

        :param int sample_size: 0 means it will extract all the data from the buffer,
            otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.

        :return: No return because all the info should be stored in the logger.
        """
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.updating = False

    @staticmethod
    def value_mask(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        """Value mask determines whether the obs_next of buffer[indices] is valid.

        For instance, usually "obs_next" after "done" flag is considered to be invalid,
        and its q/advantage value can provide meaningless (even misleading) information,
        and should be set to 0 by hand. But if "done" flag is generated because timelimit
        of game length (info["TimeLimit.truncated"] is set to True in gym's settings),
        "obs_next" will instead be valid. Value mask is typically used for assisting in
        calculating the correct q/advantage value.

        :param ReplayBuffer buffer: the corresponding replay buffer.
        :param numpy.ndarray indices: indices of replay buffer whose "obs_next" will be
            judged.

        :return: A bool type numpy.ndarray in the same shape with indices. "True" means
            "obs_next" of that buffer[indices] is valid.
        """
        return ~buffer.terminated[indices]

    @staticmethod
    def get_metrics(batch: Batch):
        cost = batch.info.get("cost", np.zeros(batch.rew.shape))
        cost = cost.astype(batch.rew.dtype)
        metrics = [batch.rew, cost]
        return metrics

    def compute_gae_returns(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        gae_lambda: float = 0.95
    ) -> Batch:
        """Compute Generalized Advantage Estimation (GAE) returns.

        This function takes in a data batch, a data buffer, an array of indices, a GAE
        lambda value, and computes the GAE returns for each critic. It returns a Batch
        object with the result stored in `batch.values`, `batch.rets`, and `batch.advs`
        as torch.Tensors.

        :param Batch batch: A data batch.
        :param ReplayBuffer buffer: The data buffer.
        :param ndarray indices: An array of indices.
        :param float gae_lambda: The GAE lambda value. Should be in [0, 1]. Defaults to
            0.95.

        :return: Batch object with the result stored in `batch.values`, `batch.rets`, and
            `batch.advs` as torch.Tensors.
        """
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        metrics = self.get_metrics(batch)
        value_mask = BasePolicy.value_mask(buffer, indices)
        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        v = [[] for _ in range(self.critics_num)]
        v_next = [[] for _ in range(self.critics_num)]
        values, returns, advantages = [], [], []

        with torch.no_grad():
            for minibatch in batch.split(
                self._max_batchsize, shuffle=False, merge_last=True
            ):
                for i in range(self.critics_num):
                    v[i].append(self.critics[i](minibatch.obs))
                    v_next[i].append(self.critics[i](minibatch.obs_next))

        for i in range(self.critics_num):
            v[i] = torch.cat(v[i], dim=0).flatten()  # old value
            values.append(v[i])
            v[i] = v[i].cpu().numpy()
            v_next[i] = torch.cat(v_next[i], dim=0).flatten().cpu().numpy()
            v_next[i] = v_next[i] * value_mask
            # when normalizing values, we do not minus self.ret_rms.mean to be
            # numerically consistent with OPENAI baselines' value normalization pipeline.
            # Emperical study also shows that "minus mean" will harm performances a tiny
            # little bit due to unknown reasons (on Mujoco envs, not confident, though).
            if self._rew_norm:  # unnormalize v_s & v_s_
                v[i] = v[i] * np.sqrt(self.ret_rms[i].var + self._eps)
                v_next[i] = v_next[i] * np.sqrt(self.ret_rms[i].var + self._eps)

            adv = gae_return(
                v[i], v_next[i], metrics[i], end_flag, self._gamma, gae_lambda
            )
            ret = adv + v[i]
            if self._rew_norm:
                ret = ret / np.sqrt(self.ret_rms[i].var + self._eps)
                self.ret_rms[i].update(ret)
            returns.append(to_torch_as(ret, values[0]))
            advantages.append(to_torch_as(adv, values[0]))

        batch.values = torch.stack(values, dim=-1)
        batch.rets = torch.stack(returns, dim=-1)
        batch.advs = torch.stack(advantages, dim=-1)
        return batch

    def compute_nstep_returns(
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        n_step: int = 1,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param ndarray indice: the sampled batch indices in the buffer.
        :param function target_q_fn: a function which compute target Q value of
            "obs_next" given data buffer and wanted indices.
        :param int n_step: the number of estimation step, should be an int greater than
            0. Default to 1.

        :return: a Batch. The result will be stored in batch.returns as a torch.Tensor
            with the same shape as target_q_fn's return tensor.
        """
        metrics = self.get_metrics(buffer)
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)  # (nstep, bsz)
        # terminal indicates buffer indexes nstep after 'indice', and are truncated at
        # the end of each episode
        terminal = indices[-1]
        # (bsz, 1)
        value_mask = BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        with torch.no_grad():
            # list of q values, each (bsz, ?)
            target_q_list = target_q_fn(buffer, terminal)
        returns = []
        for i in range(self.critics_num):
            # if i >= 1:
            #     truncation_mask = ~buffer.truncated[terminal].reshape(-1, 1)
            #     value_mask = np.logical_and(value_mask, truncation_mask)
            target_q = to_numpy(target_q_list[i].reshape(bsz, -1)) * value_mask

            target_q = nstep_return(
                metrics[i], end_flag, target_q, indices, self._gamma, n_step
            )
            returns.append(to_torch_as(target_q, target_q_list[i]))

        batch.rets = torch.stack(returns, dim=-1)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_list[0])
        return batch

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        gae_return(f64, f64, f64, b, 0.1, 0.1)
        gae_return(f32, f32, f64, b, 0.1, 0.1)
        nstep_return(f64, b, f32.reshape(-1, 1), i64, 0.1, 1)


@njit
def gae_return(
    value: np.ndarray,
    value_next: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + value_next * gamma - value
    discount = (1.0 - end_flag) * (gamma * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns


@njit
def nstep_return(
    metric: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step + 1):
        gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    target_shape = target_q.shape
    bsz = target_shape[0]
    # change target_q to 2d array
    target_q = target_q.reshape(bsz, -1)
    returns = np.zeros(target_q.shape)
    gammas = np.full(indices[0].shape, n_step)
    for n in range(n_step - 1, -1, -1):
        now = indices[n]
        gammas[end_flag[now] > 0] = n + 1
        returns[end_flag[now] > 0] = 0.0
        returns = metric[now].reshape(bsz, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    return target_q.reshape(target_shape)
