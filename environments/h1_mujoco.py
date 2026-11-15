"""Gymnasium ports of randomised-physics envs for the H1 WTE notebooks.

CheetahVel / AntDir notebooks use the hyper training envs directly.
Only CheetahDyn / WalkerDyn need this module (no HalfCheetahRandParams existed
before; WalkerDyn's legacy stack needs mujoco_py).

Physics randomization matches RandomEnv in
hyper/environments/mujoco/rand_param_envs/base.py:
body_mass / body_inertia / geom_friction ~ 1.5^u, dof_damping ~ 1.3^u,
u ~ Uniform[-L, L].
"""

from __future__ import annotations

import random

import gymnasium as gym
import mujoco
import numpy as np


def _copy_params(model):
    return {
        "body_mass": np.array(model.body_mass, copy=True),
        "body_inertia": np.array(model.body_inertia, copy=True),
        "dof_damping": np.array(model.dof_damping, copy=True),
        "geom_friction": np.array(model.geom_friction, copy=True),
    }


def _apply_params(model, data, params):
    model.body_mass[:] = params["body_mass"]
    model.body_inertia[:] = params["body_inertia"]
    model.dof_damping[:] = params["dof_damping"]
    model.geom_friction[:] = params["geom_friction"]
    mujoco.mj_setConst(model, data)


def sample_rand_params(init_params, log_scale_limit, rng=None):
    """One physics task dict, same multipliers as legacy RandomEnv."""
    rng = rng or random
    L = float(log_scale_limit)
    out = {}
    for key, base in (("body_mass", 1.5), ("body_inertia", 1.5),
                      ("geom_friction", 1.5), ("dof_damping", 1.3)):
        shape = init_params[key].shape
        u = np.array([rng.uniform(-L, L) for _ in range(int(np.prod(shape)))]).reshape(shape)
        out[key] = init_params[key] * (np.array(base) ** u)
    return out


class RandPhysEnv:
    """Gymnasium locomotion env with RandomEnv-style physics randomization."""

    def __init__(self, env_id, log_scale_limit=3.0, max_episode_steps=200):
        self.log_scale_limit = float(log_scale_limit)
        self._env = gym.make(env_id, max_episode_steps=max_episode_steps)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.init_params = _copy_params(self._env.unwrapped.model)
        self.curr_params = {k: v.copy() for k, v in self.init_params.items()}

    def sample_tasks(self, n_tasks, seed=None):
        rng = random.Random(seed) if seed is not None else random
        return [sample_rand_params(self.init_params, self.log_scale_limit, rng)
                for _ in range(n_tasks)]

    def set_task(self, task):
        _apply_params(self._env.unwrapped.model, self._env.unwrapped.data, task)
        self.curr_params = {k: np.array(v, copy=True) for k, v in task.items()}

    def reset(self, seed=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        self._env.close()


def HalfCheetahRandPhys(log_scale_limit=3.0, max_episode_steps=200):
    return RandPhysEnv("HalfCheetah-v4", log_scale_limit, max_episode_steps)


def Walker2dRandPhys(log_scale_limit=3.0, max_episode_steps=200):
    return RandPhysEnv("Walker2d-v4", log_scale_limit, max_episode_steps)
