"""Collect (s, a, r, s') tuples from GridNavi."""

import numpy as np


def collect_tuples(env, policy, num_episodes=1, max_steps=None, task=None):
    if max_steps is None:
        max_steps = env._max_episode_steps

    transitions = []
    for _ in range(num_episodes):
        env.reset_task(task=task)
        obs = env.reset()

        for _ in range(max_steps):
            action = int(policy(obs, env.get_task()))
            next_obs, reward, done, info = env.step(action)
            transitions.append((obs.copy(), action, float(reward), next_obs.copy()))
            obs = next_obs
            if done:
                break

    return transitions
