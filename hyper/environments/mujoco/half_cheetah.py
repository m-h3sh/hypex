import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_
from gymnasium.spaces import Box

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VariBAD / hyper observation: qpos[1:] (8) + qvel (9) + torso com (3) = 20
_OBS_DIM = 20


class HalfCheetahEnv(HalfCheetahEnv_):
    """HalfCheetah base on gymnasium MuJoCo, keeping the hyper/VariBAD obs layout."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float64
        )

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
            np.asarray(self.get_body_com("torso")).flat,
        ]).astype(np.float32).flatten()

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            plot_z=False,
                            **kwargs,
                            ):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        # --- roll out policy ---

        # (re)set environment
        env.reset_task()
        state, belief, task = utl.reset_env(env, args)
        start_state = state.clone()

        # keep track of what task we're in and the position of the cheetah
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        torso_index = 2 if plot_z else 0
        start_pos = unwrapped_env.get_body_com("torso")[torso_index].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if encoder is not None:
                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    prev_obs = start_state.clone()
                else:
                    prev_obs = state.clone()
                episode_prev_obs[episode_idx].append(prev_obs)
                # act
                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                _, action, _ = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)

                (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                state = state.reshape((1, -1)).float().to(device)

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[torso_index].copy())

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device), prev_obs,
                        hidden_state, return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                episode_next_obs[episode_idx].append(state.clone())
                episode_rewards[episode_idx].append(rew.clone())
                episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                if info[0]['done_mdp'] and not done:
                    start_state = info[0]['start_state']
                    start_state = torch.from_numpy(start_state).reshape((1, -1)).float().to(device)
                    start_pos = unwrapped_env.get_body_com("torso")[torso_index].copy()
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

        # clean up
        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot the movement of the half-cheetah
        plt.figure(figsize=(7, 4 * num_episodes))
        min_x = min([min(p) for p in pos])
        max_x = max([max(p) for p in pos])
        span = max_x - min_x
        for i in range(num_episodes):
            plt.subplot(num_episodes, 1, i + 1)
            # (not plotting the last step because this gives weird artifacts)
            if plot_z:
                plt.plot(range(len(pos[i][:-1])), pos[i][:-1], 'k')
                plt.title('task: {}'.format(task if args.pass_task_to_policy else env.get_task()), fontsize=15)
                plt.xlabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.ylabel('position', fontsize=15)
                plt.ylim(0, 2)
                plt.plot([200, 200], [0, 0], 'b--', alpha=0.2)
            else:
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task if args.pass_task_to_policy else env.get_task()), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

        if not return_pos:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns, pos
