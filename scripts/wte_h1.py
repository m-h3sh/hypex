"""Shared WTE helpers for the H1 notebooks (one per environment).

Pipeline (identical for every env):
  - saved row layout per transition: [ s(obs_dim) | a(a_cols) | r(1) | s'(obs_dim) ]
  - two feature sets: "sar" = (s, a, r), "sas" = (s, a, s')
  - states / next-states: per-embedding z-score (scalar mean/std over the block),
    exactly like scripts/detect.py
  - reward: pooled z-score with statistics shared across all tasks+seeds in the env
  - discrete actions one-hot; continuous actions kept raw
  - embedding: OT plan to a fixed Uniform(-1,1)^d reference (M points), then the
    centred barycentric projection (same formula as Detect.lwe), flattened
  - per-task floor F_task = mean pairwise distance of that task's embeddings across
    seeds; F (per task distribution) = mean of F_task over the TD's tasks
  - d(T) = per-seed mean pairwise distance across the TD's tasks, averaged over seeds
  - excess = d(T) - F  (reported for both L2 and cosine)
"""

from itertools import combinations

import numpy as np
import ot
import pandas as pd

METRICS = ("l2", "cosine")


def task_key(t):
    return str(tuple(t)) if hasattr(t, "__len__") else f"{float(t):.6f}"


# ----------------------------- embedding -----------------------------

def make_reference(dim, M=100, seed=98, low=-1.0, high=1.0):
    """One fixed reference cloud, reused across every task and seed."""
    return np.random.default_rng(seed).uniform(low, high, size=(M, dim))


def _zscore(block):
    block = np.asarray(block, dtype=np.float64)
    sd = block.std()
    return (block - block.mean()) / sd if sd > 0 else block - block.mean()


def build_features(X, obs_dim, a_cols, feature_set, reward_mu, reward_sd,
                   discrete=False, n_actions=None):
    X = np.asarray(X, dtype=np.float64)
    s = X[:, :obs_dim]
    a = X[:, obs_dim:obs_dim + a_cols]
    r = X[:, obs_dim + a_cols:obs_dim + a_cols + 1]
    sp = X[:, obs_dim + a_cols + 1:obs_dim + a_cols + 1 + obs_dim]
    if discrete:
        a_feat = np.zeros((len(a), n_actions))
        a_feat[np.arange(len(a)), a[:, 0].astype(int)] = 1.0
    else:
        a_feat = a
    if feature_set == "sar":
        r_z = (r - reward_mu) / reward_sd if reward_sd > 0 else r - reward_mu
        return np.concatenate([_zscore(s), a_feat, r_z], axis=1)
    if feature_set == "sas":
        return np.concatenate([_zscore(s), a_feat, _zscore(sp)], axis=1)
    raise ValueError(feature_set)


def embed(X_feat, reference, max_samples=None, rng=None):
    X_feat = np.asarray(X_feat, dtype=np.float64)
    if max_samples is not None and len(X_feat) > max_samples:
        rng = rng or np.random.default_rng(0)
        X_feat = X_feat[rng.choice(len(X_feat), max_samples, replace=False)]
    M = reference.shape[0]
    C = ot.dist(X_feat, reference)  # squared euclidean
    gamma = ot.emd(ot.unif(len(X_feat)), ot.unif(M), C, numItermax=1_000_000)
    v = (M * gamma.T @ X_feat - reference) / np.sqrt(M)
    return v.ravel()


# ----------------------------- distances -----------------------------

def _dist(u, w, metric):
    if metric == "l2":
        return float(np.linalg.norm(u - w))
    un, wn = u / np.linalg.norm(u), w / np.linalg.norm(w)
    return float(1.0 - un @ wn)


def mean_pairwise(embs, metric):
    return float(np.mean([_dist(embs[i], embs[j], metric)
                          for i, j in combinations(range(len(embs)), 2)]))


# --------------------------- orchestration ---------------------------

def pooled_reward_stats(load_fn, tasks, seeds, obs_dim, a_cols):
    """Mean/std of the reward column over all tasks+seeds in the env."""
    vals = [np.asarray(load_fn(t, s), dtype=np.float64)[:, obs_dim + a_cols]
            for s in seeds for t in tasks]
    vals = np.concatenate(vals)
    return float(vals.mean()), float(vals.std())


def compute_embeddings(task_dists, seeds, load_fn, obs_dim, a_cols, feature_set,
                       reward_mu, reward_sd, discrete=False, n_actions=None,
                       M=100, ref_seed=98, max_samples=None):
    kw = dict(obs_dim=obs_dim, a_cols=a_cols, feature_set=feature_set,
              reward_mu=reward_mu, reward_sd=reward_sd,
              discrete=discrete, n_actions=n_actions)
    sample = build_features(load_fn(next(iter(task_dists.values()))[0], seeds[0]), **kw)
    ref = make_reference(sample.shape[1], M=M, seed=ref_seed)
    emb = {}
    for td, tasks in task_dists.items():
        for t in tasks:
            for i, s in enumerate(seeds):
                feats = build_features(load_fn(t, s), **kw)
                emb[(td, task_key(t), s)] = embed(
                    feats, ref, max_samples=max_samples,
                    rng=np.random.default_rng(1000 * i + 7))
    return emb


def summarize(task_dists, seeds, emb, metrics=METRICS):
    rows = []
    for td, tasks in task_dists.items():
        row = {"TD": td, "n_tasks": len(tasks)}
        for m in metrics:
            f_task = [mean_pairwise([emb[(td, task_key(t), s)] for s in seeds], m)
                      for t in tasks]
            d_t = [mean_pairwise([emb[(td, task_key(t), s)] for t in tasks], m)
                   for s in seeds]
            F, dT = float(np.mean(f_task)), float(np.mean(d_t))
            row[f"d_{m}"] = dT
            row[f"F_{m}"] = F
            row[f"excess_{m}"] = dT - F
        rows.append(row)
    return pd.DataFrame(rows)


def _classify(nonneg, sym, collapsed, triangle):
    if nonneg and sym and triangle and not collapsed:
        return "metric"
    if nonneg and sym and triangle and collapsed:
        return "pseudometric"
    if nonneg and sym and not triangle and not collapsed:
        return "semimetric"
    return "neither"


def classify(task_dists, seeds, emb, metrics=METRICS, tol=1e-6):
    """Metric-property checks on all tasks pooled across TDs (matrix avg over seeds).

    delta (identity threshold) = mean same-task-across-seeds distance over tasks.
    """
    items = [(td, t) for td, tasks in task_dists.items() for t in tasks]
    n = len(items)
    out = []
    for m in metrics:
        delta = float(np.mean([
            mean_pairwise([emb[(td, task_key(t), s)] for s in seeds], m)
            for td, t in items]))
        D = np.zeros((n, n))
        for s in seeds:
            V = [emb[(td, task_key(t), s)] for td, t in items]
            for i in range(n):
                for j in range(i + 1, n):
                    D[i, j] = D[j, i] = D[i, j] + _dist(V[i], V[j], m)
        D /= len(seeds)
        off = D[np.triu_indices(n, k=1)]
        n_collapsed = int((off <= delta).sum())
        bad = tri = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    tri += 1
                    a, b, c = D[i, j], D[j, k], D[i, k]
                    if a + b < c - tol or a + c < b - tol or b + c < a - tol:
                        bad += 1
        out.append({
            "metric": m, "delta": delta,
            "nonneg": bool((D >= -tol).all()), "symmetric": True,
            "identity_ok": n_collapsed == 0, "n_collapsed": n_collapsed,
            "n_pairs": len(off), "triangle_ok": bad == 0,
            "triangle_bad": f"{bad}/{tri}",
            "class": _classify(True, True, n_collapsed > 0, bad == 0),
        })
    return pd.DataFrame(out)
