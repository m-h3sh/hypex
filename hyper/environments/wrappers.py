import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import gc

import torch
import torch.nn.functional as F


try:
    gym.logger.set_level(40)
except AttributeError:
    pass


class VariBadWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 episodes_per_task,
                 env_type,
                 args
                 ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - adding the done info to the state (might be needed to make states markov)
        """

        super().__init__(env)

        self.env_type = env_type
        self.args = args

        # make sure we can call these attributes even if the orig env does not have them
        if not hasattr(self.env.unwrapped, 'task_dim'):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, 'belief_dim'):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, 'get_belief'):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, 'num_states'):
            self.env.unwrapped.num_states = None
        if not hasattr(self.env.unwrapped, '_max_episode_steps'):
            self.env.unwrapped._max_episode_steps = env.max_path_length

        if episodes_per_task > 1:
            self.add_done_info = True
        else:
            self.add_done_info = False

        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box):
                if len(self.observation_space.shape) > 1:
                    raise ValueError  # can't add additional info for obs of more than 1D
                self.observation_space = spaces.Box(
                    low=np.asarray([*self.observation_space.low, 0], dtype=np.float32),
                    high=np.asarray([*self.observation_space.high, 1], dtype=np.float32),
                    dtype=np.float32,
                )
            else:
                # Not implemented. Would need to add something simliar for the other possible spaces,
                # "Space", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"
                raise NotImplementedError

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = self.episodes_per_task * self.env.unwrapped._max_episode_steps

        # add dummy timesteps in-between episodes for resetting the MDP
        self.horizon_bamdp += self.episodes_per_task - 1

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def reset(self, task=None, seed=None, options=None):
        """ Resets the BAMDP """

        try:
            self.unwrapped.reset_task(task)
        except TypeError:
            self.unwrapped.reset_task()

        # gymnasium reset returns (obs, info); training stack expects obs only
        try:
            state, _ = self.env.reset(seed=seed, options=options)
        except TypeError:
            try:
                state, _ = self.env.reset()
            except AttributeError:
                state, _ = self.unwrapped.reset()

        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, np.array([0.0], dtype=state.dtype)))

        return state

    def reset_mdp(self):
        """ Resets the underlying MDP only (*not* the task). """
        state, _ = self.env.reset()
        if self.add_done_info:
            state = np.concatenate((state, np.array([0.0], dtype=state.dtype)))
        self.done_mdp = False
        return state

    def step(self, action):

        # gymnasium: (obs, reward, terminated, truncated, info)
        state, reward, terminated, truncated, info = self.env.step(action)
        self.done_mdp = bool(terminated or truncated)

        info['done_mdp'] = self.done_mdp

        if self.add_done_info:
            state = np.concatenate((state, np.array([float(self.done_mdp)], dtype=state.dtype)))

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            if self.env_type == 'Maze':
                # In minecraft and tmaze envs, it is necessary to see start state of next ep, but not terminal state
                info['term_state'] = state
                state = self.reset_mdp()
                if self.add_done_info:
                    state[-1] = 1.0
                info['start_state'] = state
            else:
                info['start_state'] = self.reset_mdp()

        # Training stack still uses classic 4-tuple done semantics at the BAMDP level
        return state, reward, done_bamdp, info


class TimeLimitMask(gym.Wrapper):

    def step(self, action):
        # Must keep gymnasium 5-tuple: VariBadWrapper (outer) unpacks terminated/truncated.
        obs, rew, terminated, truncated, info = self.env.step(action)
        if truncated:
            info['bad_transition'] = True
        return obs, rew, terminated, truncated, info

    def reset(self, *, seed=None, options=None, **kwargs):
        # Pass through gymnasium reset; VariBadWrapper unpacks (obs, info).
        return self.env.reset(seed=seed, options=options, **kwargs)
