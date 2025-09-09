import json
import copy
import math
from collections import OrderedDict

import numpy as np
from collections.abc import Iterable

# from libero.utils.env_utils import make_libero_env
from robosuite.wrappers import Wrapper
import robosuite as suite

import torch
import os
import time
from copy import deepcopy
from easydict import EasyDict
from collections import deque
from functools import partial

import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa
import robocasa.macros as macros

import cloudpickle
import ctypes
import gym
import warnings

from abc import ABC, abstractmethod
from multiprocessing import Array, Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List, Optional, Tuple, Union

import inspect
import multiprocessing
from typing import Sequence


gym_old_venv_step_type = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)
_NP_TO_CT = {
    np.bool_: ctypes.c_bool,
    np.uint8: ctypes.c_uint8,
    np.uint16: ctypes.c_uint16,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.int8: ctypes.c_int8,
    np.int16: ctypes.c_int16,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
}


class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = "already running an async step"
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = "not running an async step"
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: (int) the number of environments
    :param observation_space: (Gym Space) the observation space
    :param action_space: (Gym Space) the action space
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: ([int] or [float]) observation
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environment's resources.
        """
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        pass

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: (Optional[int]) The random seed. May be None for completely random seeding.
        :return: (List[Union[None, int]]) Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass

    def step(self, actions):
        """
        Step the environments with the given action

        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def _get_indices(self, indices):
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: (list) the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class

    :param venv: (VecEnv) the vectorized environment to wrap
    :param observation_space: (Gym Space) the observation space (can be None to load from venv)
    :param action_space: (Gym Space) the action space (can be None to load from venv)
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def close(self):
        return self.venv.close()

    def render(self, mode: str = "human"):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def __getattr__(self, name):
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
            format_str = (
                "Error: Recursive attribute lookup for {0} from {1} is "
                "ambiguous and hides attribute from {2}"
            )
            raise AttributeError(format_str.format(name, own_class, blocked_class))

        return self.getattr_recursive(name)

    def _get_all_attributes(self):
        """Get all (inherited) instance and class attributes

        :return: (dict<str, object>) all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name):
        """Recursively check wrappers to find attribute.

        :param name (str) name of attribute to look for
        :return: (object) attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name, already_found):
        """See base class.

        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class = "{0}.{1}".format(
                type(self).__module__, type(self).__name__
            )
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     # save final observation where user can get it, then reset
                #     info['terminal_observation'] = observation
                #     observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(
                    "`{}` is not implemented in the worker".format(cmd)
                )
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # In some cases (like on GitHub workflow machine when running tests),
        # "forkserver" method results in an "connection error" (probably due to mpi)
        # We allow to bypass the default start method if an environment variable
        # is specified by the user
        if start_method is None:
            start_method = os.environ.get("DEFAULT_START_METHOD")

        # No DEFAULT_START_METHOD was specified, start_method may still be None
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = (
                "forkserver" in multiprocessing.get_all_start_methods()
            )
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe(duplex=True) for _ in range(n_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return (
            _flatten_obs(obs, self.observation_space),
            np.stack(rews),
            np.stack(dones),
            infos,
        )

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), (
        "expected list or tuple of observations per environment"
    )
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), (
            "Dict space must have ordered subspaces"
        )
        assert isinstance(obs[0], dict), (
            "non-dict observation for environment with Dict observation space"
        )
        return OrderedDict(
            [(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()]
        )
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), (
            "non-tuple observation for environment with Tuple observation space"
        )
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


GYM_RESERVED_KEYS = [
    "metadata",
    "reward_range",
    "spec",
    "action_space",
    "observation_space",
]


################################################################################
#
# Workers
#
################################################################################


class EnvWorker(ABC):
    """An abstract worker for an environment."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self._env_fn = env_fn
        self.is_closed = False
        self.result: Union[
            gym_old_venv_step_type,
            gym_new_venv_step_type,
            Tuple[np.ndarray, dict],
            np.ndarray,
        ]
        # self.action_space = self.get_env_attr("action_space")  # noqa: B009
        self.is_reset = False

    @abstractmethod
    def get_env_attr(self, key: str) -> Any:
        pass

    @abstractmethod
    def set_env_attr(self, key: str, value: Any) -> None:
        pass

    def send(self, action: Optional[np.ndarray]) -> None:
        """Send action signal to low-level worker.

        When action is None, it indicates sending "reset" signal; otherwise
        it indicates "step" signal. The paired return value from "recv"
        function is determined by such kind of different signal.
        """
        if hasattr(self, "send_action"):
            deprecation(
                "send_action will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if action is None:
                self.is_reset = True
                self.result = self.reset()
            else:
                self.is_reset = False
                self.send_action(action)

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        """Receive result from low-level worker.

        If the last "send" function sends a NULL action, it only returns a
        single observation; otherwise it returns a tuple of (obs, rew, done,
        info) or (obs, rew, terminated, truncated, info), based on whether
        the environment is using the old step API or the new one.
        """
        if hasattr(self, "get_result"):
            deprecation(
                "get_result will soon be deprecated. "
                "Please use send and recv for your own EnvWorker."
            )
            if not self.is_reset:
                self.result = self.get_result()
        return self.result

    @abstractmethod
    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        pass

    def step(
        self, action: np.ndarray
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Perform one timestep of the environment's dynamic.

        "send" and "recv" are coupled in sync simulation, so users only call
        "step" function. But they can be called separately in async
        simulation, i.e. someone calls "send" first, and calls "recv" later.
        """
        self.send(action)
        return self.recv()  # type: ignore

    @staticmethod
    def wait(
        workers: List["EnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["EnvWorker"]:
        """Given a list of workers, return those ready ones."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        # return self.action_space.seed(seed)  # issue 299
        pass

    @abstractmethod
    def render(self, **kwargs: Any) -> Any:
        """Render the environment."""
        pass

    @abstractmethod
    def close_env(self) -> None:
        pass

    def close(self) -> None:
        if self.is_closed:
            return None
        self.is_closed = True
        self.close_env()


class ShArray:
    """Wrapper of multiprocessing Array."""

    def __init__(self, dtype: np.generic, shape: Tuple[int]) -> None:
        self.arr = Array(_NP_TO_CT[dtype.type], int(np.prod(shape)))  # type: ignore
        self.dtype = dtype
        self.shape = shape

    def save(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        dst = self.arr.get_obj()
        dst_np = np.frombuffer(dst, dtype=self.dtype).reshape(self.shape)  # type: ignore
        np.copyto(dst_np, ndarray)

    def get(self) -> np.ndarray:
        obj = self.arr.get_obj()
        return np.frombuffer(obj, dtype=self.dtype).reshape(self.shape)  # type: ignore


def _setup_buf(space: gym.Space) -> Union[dict, tuple, ShArray]:
    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict)
        return {k: _setup_buf(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(space.spaces, tuple)
        return tuple([_setup_buf(t) for t in space.spaces])
    else:
        return ShArray(space.dtype, space.shape)  # type: ignore


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                p.send(obs)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class DummyEnvWorker(EnvWorker):
    """Dummy worker used in sequential vector environments."""

    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.env = env_fn()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        return getattr(self.env, key)

    def set_env_attr(self, key: str, value: Any) -> None:
        setattr(self.env.unwrapped, key, value)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        return self.env.reset(**kwargs)

    @staticmethod
    def wait(  # type: ignore
        workers: List["DummyEnvWorker"], wait_num: int, timeout: Optional[float] = None
    ) -> List["DummyEnvWorker"]:
        # Sequential EnvWorker objects are always ready
        return workers

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            self.result = self.env.reset(**kwargs)
        else:
            self.result = self.env.step(action)  # type: ignore

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        try:
            return self.env.seed(seed)  # type: ignore
        except (AttributeError, NotImplementedError):
            self.env.reset(seed=seed)
            return [seed]  # type: ignore

    def render(self, **kwargs: Any) -> Any:
        return self.env.render(**kwargs)

    def close_env(self) -> None:
        self.env.close()

    def check_success(self):
        return self.env.check_success()

    def get_segmentation_of_interest(self, segmentation_image):
        return self.env.get_segmentation_of_interest(segmentation_image)

    def get_sim_state(self):
        return self.env.get_sim_state()

    def set_init_state(self, init_state):
        return self.env.set_init_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        return self.env.regenerate_obs_from_state(mujoco_state)


class SubprocEnvWorker(EnvWorker):
    """Subprocess worker used in SubprocVectorEnv and ShmemVectorEnv."""

    def __init__(
        self, env_fn: Callable[[], gym.Env], share_memory: bool = False
    ) -> None:
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        super().__init__(env_fn)

    def get_env_attr(self, key: str) -> Any:
        self.parent_remote.send(["getattr", key])
        return self.parent_remote.recv()

    def set_env_attr(self, key: str, value: Any) -> None:
        self.parent_remote.send(["setattr", {"key": key, "value": value}])

    def _decode_obs(self) -> Union[dict, tuple, np.ndarray]:
        def decode_obs(
            buffer: Optional[Union[dict, tuple, ShArray]],
        ) -> Union[dict, tuple, np.ndarray]:
            if isinstance(buffer, ShArray):
                return buffer.get()
            elif isinstance(buffer, tuple):
                return tuple([decode_obs(b) for b in buffer])
            elif isinstance(buffer, dict):
                return {k: decode_obs(v) for k, v in buffer.items()}
            else:
                raise NotImplementedError

        return decode_obs(self.buffer)

    @staticmethod
    def wait(  # type: ignore
        workers: List["SubprocEnvWorker"],
        wait_num: int,
        timeout: Optional[float] = None,
    ) -> List["SubprocEnvWorker"]:
        remain_conns = conns = [x.parent_remote for x in workers]
        ready_conns: List[connection.Connection] = []
        remain_time, t1 = timeout, time.time()
        while len(remain_conns) > 0 and len(ready_conns) < wait_num:
            if timeout:
                remain_time = timeout - (time.time() - t1)
                if remain_time <= 0:
                    break
            # connection.wait hangs if the list is empty
            new_ready_conns = connection.wait(remain_conns, timeout=remain_time)
            ready_conns.extend(new_ready_conns)  # type: ignore
            remain_conns = [conn for conn in remain_conns if conn not in ready_conns]
        return [workers[conns.index(con)] for con in ready_conns]

    def send(self, action: Optional[np.ndarray], **kwargs: Any) -> None:
        if action is None:
            if "seed" in kwargs:
                super().seed(kwargs["seed"])
            self.parent_remote.send(["reset", kwargs])
        else:
            self.parent_remote.send(["step", action])

    def recv(
        self,
    ) -> Union[
        gym_old_venv_step_type,
        gym_new_venv_step_type,
        Tuple[np.ndarray, dict],
        np.ndarray,
    ]:  # noqa:E125
        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            if len(result) == 2:
                obs, info = result
                if self.share_memory:
                    obs = self._decode_obs()
                return obs, info
            obs = result[0]
            if self.share_memory:
                obs = self._decode_obs()
            return (obs, *result[1:])  # type: ignore
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        if "seed" in kwargs:
            super().seed(kwargs["seed"])
        self.parent_remote.send(["reset", kwargs])

        result = self.parent_remote.recv()
        if isinstance(result, tuple):
            obs, info = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs, info
        else:
            obs = result
            if self.share_memory:
                obs = self._decode_obs()
            return obs

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        super().seed(seed)
        self.parent_remote.send(["seed", seed])
        ret = self.parent_remote.recv()
        return ret

    def render(self, **kwargs: Any) -> Any:
        self.parent_remote.send(["render", kwargs])
        return self.parent_remote.recv()

    def close_env(self) -> None:
        try:
            self.parent_remote.send(["close", None])
            # mp may be deleted so it may raise AttributeError
            self.parent_remote.recv()
            self.process.join()
        except (BrokenPipeError, EOFError, AttributeError):
            pass
        # ensure the subproc is terminated
        self.process.terminate()

    def check_success(self):
        self.parent_remote.send(["check_success", None])
        return self.parent_remote.recv()

    def get_segmentation_of_interest(self, segmentation_image):
        self.parent_remote.send(["get_segmentation_of_interest", segmentation_image])
        return self.parent_remote.recv()

    def get_sim_state(self):
        self.parent_remote.send(["get_sim_state", None])
        return self.parent_remote.recv()

    def set_init_state(self, init_state):
        self.parent_remote.send(["set_init_state", init_state])
        obs = self.parent_remote.recv()
        if self.share_memory:
            obs = self._decode_obs()
        return obs


################################################################################
#
# VecEnvs
#
################################################################################


class BaseVectorEnv(object):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        worker_fn: Callable[[Callable[[], gym.Env]], EnvWorker],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._env_fns = env_fns
        # A VectorEnv contains a pool of EnvWorkers, which corresponds to
        # interact with the given envs (one worker <-> one env).
        self.workers = [worker_fn(fn) for fn in env_fns]
        self.worker_class = type(self.workers[0])
        assert issubclass(self.worker_class, EnvWorker)
        assert all([isinstance(w, self.worker_class) for w in self.workers])

        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        assert 1 <= self.wait_num <= len(env_fns), (
            f"wait_num should be in [1, {len(env_fns)}], but got {wait_num}"
        )
        self.timeout = timeout
        assert self.timeout is None or self.timeout > 0, (
            f"timeout is {timeout}, it should be positive if provided!"
        )
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn: List[EnvWorker] = []
        # environments in self.ready_id is actually ready
        # but environments in self.waiting_id are just waiting when checked,
        # and they may be ready now, but this is not known until we check it
        # in the step() function
        self.waiting_id: List[int] = []
        # all environments are ready in the beginning
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    def _assert_is_not_closed(self) -> None:
        assert not self.is_closed, (
            f"Methods of {self.__class__.__name__} cannot be called after close."
        )

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key: str) -> Any:
        """Switch the attribute getter depending on the key.

        Any class who inherits ``gym.Env`` will inherit some attributes, like
        ``action_space``. However, we would like the attribute lookup to go straight
        into the worker (in fact, this vector env's action_space is always None).
        """
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return self.get_env_attr(key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        """Get an attribute from the underlying environments.

        If id is an int, retrieve the attribute denoted by key from the environment
        underlying the worker at index id. The result is returned as a list with one
        element. Otherwise, retrieve the attribute for all workers at indices id and
        return a list that is ordered correspondingly to id.

        :param str key: The key of the desired attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.

        :return list: The list of environment attributes.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        return [self.workers[j].get_env_attr(key) for j in id]

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        """Set an attribute in the underlying environments.

        If id is an int, set the attribute denoted by key from the environment
        underlying the worker at index id to value.
        Otherwise, set the attribute for all workers at indices id.

        :param str key: The key of the desired attribute.
        :param Any value: The new value of the attribute.
        :param id: Indice(s) of the desired worker(s). Default to None for all env_id.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for j in id:
            self.workers[j].set_env_attr(key, value)

    def _wrap_id(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[List[int], np.ndarray]:
        if id is None:
            return list(range(self.env_num))
        return [id] if np.isscalar(id) else id  # type: ignore

    def _assert_id(self, id: Union[List[int], np.ndarray]) -> None:
        for i in id:
            assert i not in self.waiting_id, (
                f"Cannot interact with environment {i} which is stepping now."
            )
            assert i in self.ready_id, (
                f"Can only interact with ready environments {self.ready_id}."
            )

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, **kwargs)
        ret_list = [self.workers[i].recv() for i in id]

        reset_returns_info = (
            isinstance(ret_list[0], (tuple, list))
            and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )
        if reset_returns_info:
            obs_list = [r[0] for r in ret_list]
        else:
            obs_list = ret_list

        if isinstance(obs_list[0], tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        if reset_returns_info:
            infos = [r[1] for r in ret_list]
            return obs, infos  # type: ignore
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Union[gym_old_venv_step_type, gym_new_venv_step_type]:
        """Run one timestep of some environments' dynamics.

        If id is None, run one timestep of all the environments’ dynamics;
        otherwise run one timestep for some environments with given id,  either
        an int or a list. When the end of episode is reached, you are
        responsible for calling reset(id) to reset this environment’s state.

        Accept a batch of action and return a tuple (batch_obs, batch_rew,
        batch_done, batch_info) in numpy format.

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple consisting of either:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            or:

            * ``obs`` a numpy.ndarray, the agent's observation of current environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``terminated`` a numpy.ndarray, whether these episodes have been \
                terminated
            * ``truncated`` a numpy.ndarray, whether these episodes have been truncated
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)

            The case distinction is made based on whether the underlying environment
            uses the old step API (first case) or the new step API (second case).

        For the async simulation:

        Provide the given action to the environments. The action sequence
        should correspond to the ``id`` argument, and the ``id`` argument
        should be a subset of the ``env_id`` in the last returned ``info``
        (initially they are env_ids of all the environments). If action is
        None, fetch unfinished step() calls instead.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if not self.is_async:
            assert len(action) == len(id)
            for i, j in enumerate(id):
                self.workers[j].send(action[i])
            result = []
            for j in id:
                env_return = self.workers[j].recv()
                env_return[-1]["env_id"] = j
                result.append(env_return)
        else:
            if action is not None:
                self._assert_id(id)
                assert len(action) == len(id)
                for act, env_id in zip(action, id):
                    self.workers[env_id].send(act)
                    self.waiting_conn.append(self.workers[env_id])
                    self.waiting_id.append(env_id)
                self.ready_id = [x for x in self.ready_id if x not in id]
            ready_conns: List[EnvWorker] = []
            while not ready_conns:
                ready_conns = self.worker_class.wait(
                    self.waiting_conn, self.wait_num, self.timeout
                )
            result = []
            for conn in ready_conns:
                waiting_index = self.waiting_conn.index(conn)
                self.waiting_conn.pop(waiting_index)
                env_id = self.waiting_id.pop(waiting_index)
                # env_return can be (obs, reward, done, info) or
                # (obs, reward, terminated, truncated, info)
                env_return = conn.recv()
                env_return[-1]["env_id"] = env_id  # Add `env_id` to info
                result.append(env_return)
                self.ready_id.append(env_id)
        return_lists = tuple(zip(*result))
        obs_list = return_lists[0]
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs_stack = np.array(obs_list, dtype=object)
        other_stacks = map(np.stack, return_lists[1:])
        return (obs_stack, *other_stacks)  # type: ignore

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        return [w.seed(s) for w, s in zip(self.workers, seed_list)]

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self._assert_is_not_closed()
        if self.is_async and len(self.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.waiting_id} are still stepping, cannot "
                "render them now."
            )
        return [w.render(**kwargs) for w in self.workers]

    def close(self) -> None:
        """Close all of the environments.

        This function will be called only once (if not, it will be called during
        garbage collected). This way, ``close`` of all workers can be assured.
        """
        self._assert_is_not_closed()
        for w in self.workers:
            w.close()
        self.is_closed = True


class DummyVectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, DummyEnvWorker, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs

    def regenerate_obs_from_state(self, mujoco_state):
        self._assert_is_not_closed()
        id = self._wrap_id(None)
        if self.is_async:
            self._assert_id(id)

        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].regenerate_obs_from_state(mujoco_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for other APIs' usage.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> SubprocEnvWorker:
            return SubprocEnvWorker(fn, share_memory=False)

        super().__init__(env_fns, worker_fn, **kwargs)

    def check_success(self):
        return [w.check_success() for w in self.workers]

    def get_segmentation_of_interest(self, segmentation_images):
        return [
            w.get_segmentation_of_interest(img)
            for w, img in zip(self.workers, segmentation_images)
        ]

    def get_sim_state(self):
        return [w.get_sim_state() for w in self.workers]

    def set_init_state(
        self,
        init_state: Optional[Union[int, List[int], np.ndarray]] = None,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Reset the state of some envs and return initial observations.
        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        # send(None) == reset() in worker
        obs_list = []
        for j, i in enumerate(id):
            obs = self.workers[i].set_init_state(init_state[j])
            obs_list.append(obs)
        obs = np.stack(obs_list)
        return obs


def merge_dict(dict_obj):
    merged_dict = {}
    for k in dict_obj[0].keys():
        merged_dict[k] = np.stack([d[k] for d in dict_obj], axis=0)
    return merged_dict


class StackDummyVectorEnv(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, id=None):
        obs = super().reset(id=id)
        return merge_dict(obs)

    def step(self, action: np.ndarray, id=None):
        obs, reward, done, info = super().step(action, id)
        return merge_dict(obs), reward, done, merge_dict(info)


class StackSubprocVectorEnv(SubprocVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        return merge_dict(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return merge_dict(obs), reward, done, merge_dict(info)


class TaskEmbWrapper(Wrapper):
    """Wrapper to add task embeddings to the returned info"""

    def __init__(self, env, task_emb):
        super().__init__(env)
        self.task_emb = task_emb

    def reset(self):
        obs = self.env.reset()
        obs["task_emb"] = self.task_emb
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["task_emb"] = self.task_emb
        return obs, reward, done, info


class RoboCasaResetWrapper(Wrapper):
    """Wrap reset with predefined initial states"""

    def __init__(self, env, init_states):
        super().__init__(env)
        self.init_states = init_states
        self.reset_times = 0

    def reset(self):
        _ = self.env.reset()
        if len(self.init_states) > 0:
            obs = self.env.set_init_state(self.init_states[self.reset_times])
        else:
            obs = self.env.reset()

        self.reset_times += 1
        if self.reset_times == len(self.init_states):
            self.reset_times = 0
        return obs

    def seed(self, seed):
        self.env.seed(seed)


def make_robocasa_env(
    task_name,
    img_h,
    img_w,
    task_embedding=None,
    gpu_id=-1,
    vec_env_num=1,
    seed=0,
    init_states=None,
):
    env_args = {
        "env_name": task_name,
        "robots": ["PandaOmron"],
        "render_camera": None,
        "camera_names": ["robot0_agentview_right", "robot0_eye_in_hand"],
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "camera_heights": img_h,
        "camera_widths": img_w,
        "render_gpu_device_id": gpu_id,
    }

    if task_embedding is None:
        task_embs = torch.zeros(1, 768)
    else:
        task_embs = task_embedding

    assert init_states is None or len(init_states) % vec_env_num == 0, (
        "error: number of initial states must be divisible by number of envs"
    )

    def env_func(env_idx):
        base_env = ControlEnv(**env_args)
        base_env = TaskEmbWrapper(base_env, task_emb=task_embs[0])
        # TODO: should seed
        return base_env

    envs = [partial(env_func, env_idx=i) for i in range(vec_env_num)]
    env = StackDummyVectorEnv(envs) if vec_env_num == 1 else StackSubprocVectorEnv(envs)

    return env, task_embs


class ObservationWrapper(Wrapper):
    valid_obs_types = ["image"]

    def __init__(self, env, masks, cameras):
        super(ObservationWrapper, self).__init__(env)
        self.masks = masks
        self.cameras = cameras

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._stack_obs(obs)
        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self._stack_obs(obs)
        return obs_dict, reward, done, info

    def _stack_obs(self, obs):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=0)
        return obs_dict


class LiberoObservationWrapper(ObservationWrapper):
    valid_obs_types = ["image"]

    def __init__(self, env, masks, cameras):
        super().__init__(env, masks, cameras)

    def _stack_obs(self, obs, axis=0):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=1)  # (b, v, h, w, c)
        return obs_dict


class LiberoSuccessWrapper(Wrapper):
    def __init__(self, env):
        super(LiberoSuccessWrapper, self).__init__(env)
        self.success = None

    def reset(self):
        obs = self.env.reset()
        self.success = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.success is None:
            self.success = done
        else:
            assert len(self.success) == len(done)
            self.success = [self.success[i] or done[i] for i in range(len(done))]
        info["success"] = list(self.success)
        return obs, reward, done, info


class ControlEnv:
    def __init__(
        self,
        env_name="PnPCounterToStove",
        robots=["PandaOmron"],
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "robot0_agentview_right",
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        controller_config = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )
        config = {
            "env_name": env_name,
            "robots": robots,
            "controller_configs": controller_config,
        }
        self.env = robosuite.make(
            **config,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="robot0_frontview",
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            ignore_done=True,
            use_camera_obs=True,
            control_freq=20,
            renderer="mjviewer",
        )
        self.observation_space, self.action_space = self._get_obs_action_spaces()

    def _get_obs_action_spaces(self):
        obs = self.reset()
        obs_space = {k: np.zeros_like(v) for k, v in obs.items()}
        act_space = -np.ones((1, 7))  # TODO: currently hardcoded for Panda
        return obs_space, act_space

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        # append 0, 0, 0, 0, -1 to action
        action = np.concatenate([action, np.array([0, 0, 0, 0, -1])], axis=-1)
        return self.env.step(action)

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


def build_env(
    img_size,
    env_type,
    env_meta_fn=None,
    env_name=None,
    task_name=None,
    render_gpu_ids=-1,
    vec_env_num=1,
    seed=0,
    env_idx_start_end=None,
    **kwargs,
):
    """
    Build the rollout environment.
    Args:
        img_size: The resolution of the pixel observation.
        env_type: The type of environment benchmark. Choices: ["libero"].
        env_meta_fn: The path to robommimic meta data, which is used to specify the robomimic environments.
        env_name: The name to specify the environments.
        obs_types: The observation types in the returned obs dict in Robomimic
        render_gpu_ids:  The available GPU ids for rendering the images
        vec_env_num: The number of parallel environments
        seed: The random seed environment initialization.

    Returns:
        env: A gym-like environment.
    """
    if isinstance(img_size, Iterable):
        assert len(img_size) == 2
        img_h = img_size[0]
        img_w = img_size[1]
    else:
        img_h = img_w = img_size

    if env_type.lower() == "libero":
        if isinstance(render_gpu_ids, Iterable):
            render_gpu_ids = [int(i) for i in render_gpu_ids]
            gpu_id_for_each_env = render_gpu_ids * math.ceil(
                len(env_name) / len(render_gpu_ids)
            )
            gpu_id_for_each_env = gpu_id_for_each_env[: len(env_name)]
        else:
            gpu_id_for_each_env = [render_gpu_ids] * len(env_name)

        if env_idx_start_end is not None:
            idx_start, idx_end = env_idx_start_end
        else:
            idx_start = 0
            idx_end = len(env_name)

        env_dict = OrderedDict()
        suite_to_task_embs = {}
        for env_idx in range(idx_start, idx_end):
            e_name, t_name, e_meta_fn, gpu_id = (
                env_name[env_idx],
                task_name[env_idx],
                env_meta_fn[env_idx],
                gpu_id_for_each_env[env_idx],
            )
            task_embedding = suite_to_task_embs.get(e_name, None)
            env, task_embs = make_robocasa_env(
                t_name,
                img_h,
                img_w,
                task_embedding=task_embedding,
                gpu_id=gpu_id,
                vec_env_num=vec_env_num,
                seed=seed,
            )
            if e_name not in suite_to_task_embs:
                suite_to_task_embs[e_name] = task_embs
            env = LiberoSuccessWrapper(env)
            with open(e_meta_fn) as f:
                env_meta = json.load(f)
            cameras = ["robot0_eye_in_hand", "robot0_agentview_right"]
            cameras.sort()
            env = LiberoObservationWrapper(env, masks=None, cameras=cameras)
            env_dict[f"{e_name}/{t_name}"] = (env_idx, env)
        env = env_dict
    else:
        raise ValueError(f"Environment {env_type} is not supported!")
    return env
