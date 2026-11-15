"""Gymnasium ports of H1 MuJoCo envs (env.yml only; no mujoco_py).

CheetahVel / AntDir mirror hyper's reward, obs, and termination as closely as
the gymnasium MuJoCo models allow (same formulas; modern free MuJoCo backend).

Dyn envs: RandomEnv-style physics scaling from
hyper/environments/mujoco/rand_param_envs/base.py.
"""

from __future__ import annotations

import random

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box


# ---------------------------------------------------------------------------
# HalfCheetahVel — matches hyper/environments/mujoco/half_cheetah_vel.py
# Obs: qpos[1:] (8) + qvel (9) + torso com (3) = 20  (same as hyper HalfCheetahEnv)
# Reward: -|v - v*| - 0.05 ||a||^2 ; never terminates (only time-limit truncate)
# ---------------------------------------------------------------------------

class HalfCheetahVelEnv:
    def __init__(self, max_episode_steps=200, vel_low=0.0, vel_high=3.0):
        self.vel_low = float(vel_low)
        self.vel_high = float(vel_high)
        self._max_episode_steps = int(max_episode_steps)
        self._env = gym.make("HalfCheetah-v4", max_episode_steps=self._max_episode_steps)
        self.unwrapped = self._env.unwrapped
        self.action_space = self._env.action_space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float64)
        self.goal_velocity = float(self.sample_tasks(1)[0])

    def _get_obs(self):
        u = self.unwrapped
        return np.concatenate([
            u.data.qpos.flat[1:],
            u.data.qvel.flat,
            np.asarray(u.get_body_com("torso")).flat,
        ]).astype(np.float64).ravel()

    def set_task(self, task):
        self.goal_velocity = float(task)

    def get_task(self):
        return np.array([self.goal_velocity])

    def sample_tasks(self, n_tasks):
        return [random.uniform(self.vel_low, self.vel_high) for _ in range(n_tasks)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def reset(self, seed=None):
        self._env.reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        u = self.unwrapped
        xpos_before = float(u.data.qpos[0])
        # advance physics (ignore gymnasium reward / termination)
        u.do_simulation(action, u.frame_skip)
        xpos_after = float(u.data.qpos[0])
        forward_vel = (xpos_after - xpos_before) / u.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * float(np.sum(np.square(action)))
        reward = forward_reward - ctrl_cost
        # time limit via wrapper
        self._env._elapsed_steps = getattr(self._env, "_elapsed_steps", 0) + 1
        truncated = self._env._elapsed_steps >= self._max_episode_steps
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost,
                     task=self.get_task())
        return self._get_obs(), reward, False, truncated, infos

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# AntDir — matches hyper/environments/mujoco/ant_dir.py
# Obs: qpos[2:] + qvel  (hyper AntEnv._get_obs)
# Reward: dir·v_xy - 0.5||a||^2 - contact + 1.0 survive
# Done when height leaves [0.2, 1.0] or non-finite
# ---------------------------------------------------------------------------

class AntDirEnv:
    def __init__(self, max_episode_steps=200):
        self._max_episode_steps = int(max_episode_steps)
        self._env = gym.make("Ant-v4", max_episode_steps=self._max_episode_steps)
        self.unwrapped = self._env.unwrapped
        self.action_space = self._env.action_space
        # hyper-style obs size (same length as stock Ant-v4, different contents)
        o0 = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=o0.shape, dtype=np.float64)
        self.goal_direction = 0.0

    def _get_obs(self):
        u = self.unwrapped
        return np.concatenate([
            u.data.qpos.flat[2:],
            u.data.qvel.flat,
        ]).astype(np.float64).ravel()

    def set_task(self, task):
        self.goal_direction = float(task)

    def get_task(self):
        return np.array([self.goal_direction])

    def sample_tasks(self, n_tasks):
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks)]

    def reset(self, seed=None):
        self._env.reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        u = self.unwrapped
        torso_before = np.array(u.get_body_com("torso"), copy=True)
        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction))
        u.do_simulation(action, u.frame_skip)
        torso_after = np.array(u.get_body_com("torso"), copy=True)
        torso_velocity = torso_after - torso_before
        forward_reward = float(np.dot(torso_velocity[:2] / u.dt, direct))
        ctrl_cost = 0.5 * float(np.sum(np.square(action)))
        contact_cost = 0.5 * 1e-3 * float(np.sum(np.square(np.clip(u.data.cfrc_ext, -1, 1))))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = u.state_vector()
        terminated = not (np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
        self._env._elapsed_steps = getattr(self._env, "_elapsed_steps", 0) + 1
        truncated = (not terminated) and (self._env._elapsed_steps >= self._max_episode_steps)
        infos = dict(
            reward_forward=forward_reward, reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost, reward_survive=survive_reward,
            torso_velocity=torso_velocity, task=self.get_task(),
        )
        return self._get_obs(), reward, terminated, truncated, infos

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Randomised physics (CheetahDyn / WalkerDyn)
# ---------------------------------------------------------------------------

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
