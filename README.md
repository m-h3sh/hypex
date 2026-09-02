# hypex

Exploring the role of hypernetworks in few-shot meta reinforcement learning.

This repo currently focuses on collecting GridWorld trajectories and computing Wasserstein Task Embeddings (WTE) to compare task similarity under different goal distributions.

## Setup

Create and activate the conda environment from `env.yml`:

```bash
conda env create -f env.yml
conda activate adv_rl
```

To refresh packages later:

```bash
conda env update -f env.yml --prune
```

The main notebook is `scripts/exp1.ipynb`.

## Layout

| Path | Contents |
|------|----------|
| `env.yml` | Conda env (`adv_rl`): Python 3.10, PyTorch, POT, gym, sklearn, etc. |
| `environments/` | `GridNavi` grid navigation Gym env (`gridworld.py`). |
| `scripts/` | Experiment notebook and embedding / plotting helpers. |
| `data/` | Saved transition arrays per task distribution (`task_dist1/2/3`). |
| `src/` | Reserved for later source modules. |

### Scripts

| File | Role |
|------|------|
| `scripts/exp1.ipynb` | Main experiment (see below). |
| `scripts/detect.py` | `Detect` class: preprocess transitions, build a random reference set, compute LWE / WTE embeddings via optimal transport. |
| `scripts/collect.py` | Helper to roll out an env/policy and collect `(s, a, r, s')` tuples. |
| `scripts/dissimilarity.py` | Older pairwise embedding-distance helper (related to Detect). |
| `scripts/plotting.py` | Heatmap utilities used for similarity matrices. |

## Experiment: `scripts/exp1.ipynb`

End-to-end pipeline for comparing goal distributions with WTE:

1. **Imports / path setup** — loads `GridNavi`, `Detect`, and plotting helpers.
2. **Env smoke test** — builds a small `GridNavi` and optionally visualizes a random episode.
3. **Task distributions** — defines three goal sets on a `NUM_CELLS × NUM_CELLS` grid:
   - `task_dist1`: goals clustered in the top-right corner
   - `task_dist2`: more spread-out goals
   - `task_dist3`: four corners
4. **Data collection** — for each goal, samples random actions and saves `N` rows of `[x, y, a, r, x', y']` under `data/task_dist{k}/goal_{x}_{y}.npy`.
5. **Embeddings** — instantiates `Detect`, sets a shared random reference, and computes an LWE embedding per saved goal file.
6. **Similarity heatmaps** — pairwise L2 distances between embeddings within each distribution (MOSAIC-style visualization).

Run cells top-to-bottom. Re-run the collection cell only when you want to regenerate `data/`; otherwise load existing `.npy` files and jump to the embedding / heatmap cells.
