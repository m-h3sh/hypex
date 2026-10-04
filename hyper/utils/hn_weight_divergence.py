"""
Online probe: accumulate HN-generated *actor* (policy) weights at BAMDP meta-episode ends,
then flush mean pairwise L2 (+ mean vectors) every log_interval.

Critic weights are excluded — we only care about policy change across tasks.
"""
import os

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HNWeightDivergenceTracker:
    def __init__(self, args, logger, actor_critic, num_cells):
        self.args = args
        self.logger = logger
        self.actor_critic = actor_critic
        self.num_cells = int(num_cells)
        self.enabled = bool(getattr(args, "log_hn_weight_div", False) and args.use_hypernet)
        self._sum = {}    # cell_id -> CPU tensor
        self._count = {}  # cell_id -> int

    def _goal_to_id(self, goal):
        g = np.asarray(goal).reshape(-1)
        gx, gy = int(g[0]), int(g[1])
        return gx * self.num_cells + gy

    def _weights_from_latent(self, latent):
        """Flattened actor-only HN base params for one latent (no critic)."""
        ac = self.actor_critic
        z = latent.detach()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = z.to(device)
        bsz = z.shape[0]
        state = torch.zeros(bsz, ac.dim_state, device=device)
        # belief/task unused for hypernet_input=latent; Policy replaces None with empty tensors
        state_e, latent_e, belief_e, task_e, one_hot_task, _ = ac.get_encoded_inputs(
            state, z, None, None
        )
        if ac.args.hypernet_input == "latent":
            inputs = torch.cat((state_e, belief_e, task_e), dim=-1)
            hyper_input = latent_e
        else:
            assert ac.args.hypernet_input == "task_embed", ac.args.hypernet_input
            inputs = torch.cat((state_e, latent_e, belief_e), dim=-1)
            hyper_input = task_e
        _, a_base_params = ac.forward_actor(
            inputs,
            hyper_input=hyper_input,
            return_base_params=True,
            one_hot_tasks=one_hot_task,
            training=False,
        )
        return torch.cat([p.reshape(bsz, -1) for p in a_base_params], dim=-1).squeeze(0).cpu()

    def record(self, latent, goal):
        if not self.enabled:
            return
        with torch.no_grad():
            theta = self._weights_from_latent(latent)
        tid = self._goal_to_id(goal)
        if tid not in self._sum:
            self._sum[tid] = torch.zeros_like(theta)
            self._count[tid] = 0
        self._sum[tid] += theta
        self._count[tid] += 1

    def flush(self, iter_idx):
        if not self.enabled:
            return None
        if len(self._sum) < 2:
            self._sum.clear()
            self._count.clear()
            return None

        means = {}
        for tid, s in self._sum.items():
            means[tid] = s / max(self._count[tid], 1)

        ids = sorted(means.keys())
        vecs = torch.stack([means[i] for i in ids], dim=0)
        # mean pairwise L2 over unique pairs
        diffs = vecs.unsqueeze(0) - vecs.unsqueeze(1)  # (T, T, D)
        dists = torch.linalg.vector_norm(diffs, dim=-1)
        # upper triangle only
        iu = torch.triu_indices(len(ids), len(ids), offset=1)
        mean_pairwise = dists[iu[0], iu[1]].mean().item()

        if self.logger is not None:
            self.logger.add("hn_weight_div/mean_pairwise_l2", mean_pairwise, iter_idx)
            out_dir = os.path.join(self.logger.full_output_folder, "hn_weight_div")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "means": {int(k): v for k, v in means.items()},
                    "counts": {int(k): int(self._count[k]) for k in means},
                    "num_cells": self.num_cells,
                    "iter": int(iter_idx),
                    "mean_pairwise_l2": mean_pairwise,
                },
                os.path.join(out_dir, f"iter_{iter_idx}.pt"),
            )

        self._sum.clear()
        self._count.clear()
        return mean_pairwise
