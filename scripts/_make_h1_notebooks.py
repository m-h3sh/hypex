"""Generate one H1 WTE notebook per environment. All share wte_h1.py and the
same analysis/classification cells; only the config+collection cell differs."""
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# ----- cells shared by every notebook -----

COLLECT_RUN = """\
# Collect (s,a,r,s') for every task over all seeds. Safe to re-run: skips files.
for _reward in REWARDS:
    for _seed in SEEDS:
        collect(_reward, _seed)
print("collection done:", {r: len(SEEDS) for r in REWARDS}, "seeds x",
      sum(len(v) for v in TASK_DISTS.values()), "tasks")"""

ANALYSIS = """\
# Embed, then per TD: F (mean per-task floor), d(T) (mean pairwise over seeds),
# and excess = d(T) - F, for both L2 and cosine, for (s,a,r) and (s,a,s').
EMB, _tables = {}, []
for _reward in REWARDS:
    _load = load_fn_for(_reward)
    _mu, _sd = h1.pooled_reward_stats(_load, ALL_TASKS, SEEDS,
                                      SPEC["obs_dim"], SPEC["a_cols"])
    for _fs in FEATURE_SETS:
        _emb = h1.compute_embeddings(
            TASK_DISTS, SEEDS, _load, SPEC["obs_dim"], SPEC["a_cols"], _fs, _mu, _sd,
            discrete=SPEC["discrete"], n_actions=SPEC["n_actions"],
            max_samples=MAX_SAMPLES)
        EMB[(_reward, _fs)] = _emb
        _t = h1.summarize(TASK_DISTS, SEEDS, _emb)
        _t.insert(0, "features", _fs)
        _t.insert(0, "reward", _reward)
        _tables.append(_t)
results = pd.concat(_tables, ignore_index=True)
results.round(4)"""

CLASSIFY = """\
# Metric-property checks on all tasks pooled across TDs (pairwise matrix averaged
# over seeds; identity threshold delta = mean same-task-across-seeds distance).
_cls = [h1.classify(TASK_DISTS, SEEDS, _emb).assign(reward=_r, features=_f)
        for (_r, _f), _emb in EMB.items()]
classification = pd.concat(_cls, ignore_index=True)[[
    "reward", "features", "metric", "delta", "nonneg", "symmetric",
    "identity_ok", "n_collapsed", "n_pairs", "triangle_ok", "triangle_bad", "class"]]
classification.round(4)"""

# ----- per-env config + collection cells -----

GRID_CFG = '''\
import sys
from pathlib import Path
import numpy as np, pandas as pd

CODE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
for _p in (str(CODE_DIR), str(CODE_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from environments.gridworld import GridNavi
import wte_h1 as h1

NUM_CELLS, MAX_STEPS = 6, 20
N = NUM_CELLS * NUM_CELLS * MAX_STEPS          # 720 tuples / goal
SEEDS = list(range(42, 52))                    # 10 data seeds
SPEC = dict(obs_dim=2, a_cols=1, discrete=True, n_actions=5)
FEATURE_SETS = ["sar", "sas"]
REWARDS = ["distance", "sparse"]               # grid has two reward variants
MAX_SAMPLES = None
DATA = CODE_DIR / "data" / "grid6"

def _g(opt):
    c = NUM_CELLS
    return {1: [(c-1, c-1), (c-1, c-2), (c-2, c-1), (c-2, c-2)],   # adjacent
            2: [(c//2, c//2), (c-1, c//2), (c//2, c-1), (c-1, c-1)],  # mid/edge
            3: [(0, 0), (0, c-1), (c-1, 0), (c-1, c-1)]}[opt]        # corners
TASK_DISTS = {1: _g(1), 2: _g(2), 3: _g(3)}
ALL_TASKS = [t for ts in TASK_DISTS.values() for t in ts]

def _dir(reward, seed):
    return DATA / reward / f"seed{seed}"

def collect(reward, seed):
    out = _dir(reward, seed); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); dr = reward == "distance"
    for tasks in TASK_DISTS.values():
        for task in tasks:
            f = out / f"goal_{task[0]}_{task[1]}.npy"
            if f.exists():
                continue
            rows = []
            for j in range(NUM_CELLS):
                for k in range(NUM_CELLS):
                    env = GridNavi(num_cells=NUM_CELLS, num_steps=MAX_STEPS,
                                   distance_reward=dr, starting_state=(j, k),
                                   possible_goals=list(tasks))
                    env.reset_task(task=task); obs, _ = env.reset()
                    for _ in range(MAX_STEPS):
                        a = int(rng.integers(env.action_space.n))
                        nobs, r, term, trunc, _ = env.step(a)
                        rows.append([obs[0], obs[1], a, r, nobs[0], nobs[1]]); obs = nobs
                        if term or trunc:
                            obs, _ = env.reset()
            np.save(f, np.asarray(rows[:N]))

def load_fn_for(reward):
    def load(task, seed):
        return np.load(_dir(reward, seed) / f"goal_{task[0]}_{task[1]}.npy")
    return load'''

# All MuJoCo H1 notebooks use environments/h1_mujoco.py (gymnasium / env.yml).

CHEVEL_CFG = '''\
import sys
from pathlib import Path
import numpy as np, pandas as pd

CODE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
for _p in (str(CODE_DIR), str(CODE_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from environments.h1_mujoco import HalfCheetahVelEnv  # mirrors hyper HalfCheetahVelEnv
import wte_h1 as h1

EP_LEN, N_EPISODES = 200, 25
N = EP_LEN * N_EPISODES                          # ~5000 tuples / task
SEEDS = list(range(42, 47))                      # 5 data seeds
FEATURE_SETS = ["sar", "sas"]
REWARDS = ["default"]
MAX_SAMPLES = None                               # set e.g. 3000 to speed up OT
DATA = CODE_DIR / "data" / "chevel_h1"

VEL_RANGES = {1: (0.5, 1.0), 2: (0.75, 2.25), 3: (0.0, 3.0)}   # report training TDs
N_TASKS = 8
TASK_DISTS = {k: [round(float(v), 4) for v in np.linspace(lo, hi, N_TASKS)]
              for k, (lo, hi) in VEL_RANGES.items()}
ALL_TASKS = [t for ts in TASK_DISTS.values() for t in ts]

_probe = HalfCheetahVelEnv(max_episode_steps=EP_LEN)
_o, _ = _probe.reset()
SPEC = dict(obs_dim=int(np.ravel(_o).shape[0]),   # 20, same as hyper
            a_cols=int(_probe.action_space.shape[0]),
            discrete=False, n_actions=None)
_probe.close()

def _dir(reward, seed):
    return DATA / f"seed{seed}"

def collect(reward, seed):
    out = _dir(reward, seed); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = HalfCheetahVelEnv(max_episode_steps=EP_LEN)
    for vel in ALL_TASKS:
        f = out / f"vel_{vel:.4f}.npy"
        if f.exists():
            continue
        env.reset_task(vel); obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        rows = []
        while len(rows) < N:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            nobs, r, term, trunc, _ = env.step(a)
            rows.append(np.concatenate([np.ravel(obs), np.ravel(a), [r], np.ravel(nobs)]))
            obs = nobs
            if term or trunc:
                env.reset_task(vel); obs, _ = env.reset()
        np.save(f, np.asarray(rows[:N], dtype=np.float64))
    env.close()

def load_fn_for(reward):
    def load(task, seed):
        return np.load(_dir(reward, seed) / f"vel_{task:.4f}.npy")
    return load'''

ANTDIR_CFG = '''\
import sys
from pathlib import Path
import numpy as np, pandas as pd

CODE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
for _p in (str(CODE_DIR), str(CODE_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from environments.h1_mujoco import AntDirEnv  # mirrors hyper AntDirEnv (no mujoco_py)
import wte_h1 as h1

EP_LEN = 200
N = 5000                                         # tuples / task
SEEDS = list(range(42, 47))                      # 5 data seeds
FEATURE_SETS = ["sar", "sas"]
REWARDS = ["default"]
MAX_SAMPLES = None
DATA = CODE_DIR / "data" / "antdir_h1"

N_TASKS = 8
TASK_DISTS = {
    1: [float(a) for a in np.linspace(np.deg2rad(-15), np.deg2rad(15), N_TASKS)],
    2: [float(a) for a in np.linspace(np.deg2rad(-90), np.deg2rad(90), N_TASKS)],
    3: [float(a) for a in np.linspace(-np.pi, np.pi, N_TASKS, endpoint=False)],
}
ALL_TASKS = [t for ts in TASK_DISTS.values() for t in ts]

_probe = AntDirEnv(max_episode_steps=EP_LEN)
_o, _ = _probe.reset()
SPEC = dict(obs_dim=int(np.ravel(_o).shape[0]),   # 27, same layout as hyper Ant
            a_cols=int(_probe.action_space.shape[0]),
            discrete=False, n_actions=None)
_probe.close()

def _dir(reward, seed):
    return DATA / f"seed{seed}"

def collect(reward, seed):
    out = _dir(reward, seed); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = AntDirEnv(max_episode_steps=EP_LEN)
    for ang in ALL_TASKS:
        f = out / f"dir_{ang:.6f}.npy"
        if f.exists():
            continue
        env.set_task(ang); obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        rows = []
        while len(rows) < N:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            nobs, r, term, trunc, _ = env.step(a)
            rows.append(np.concatenate([np.ravel(obs), np.ravel(a), [r], np.ravel(nobs)]))
            obs = nobs
            if term or trunc:
                env.set_task(ang); obs, _ = env.reset()
        np.save(f, np.asarray(rows[:N], dtype=np.float64))
    env.close()

def load_fn_for(reward):
    def load(task, seed):
        return np.load(_dir(reward, seed) / f"dir_{task:.6f}.npy")
    return load'''

# Physics: fixed param dicts sampled once; only the rollout RNG changes per seed.
PHYS_CFG_TMPL = '''\
import sys
from pathlib import Path
import numpy as np, pandas as pd

CODE_DIR = Path.cwd().parent if Path.cwd().name == "scripts" else Path.cwd()
for _p in (str(CODE_DIR), str(CODE_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from environments.h1_mujoco import {FACTORY}
import wte_h1 as h1

N = 5000                                         # tuples / task
SEEDS = list(range(42, 47))                      # 5 data seeds
FEATURE_SETS = ["sar", "sas"]                    # (s,a,s') carries the signal here
REWARDS = ["default"]
MAX_SAMPLES = None
DATA = CODE_DIR / "data" / "{DATA_NAME}"

L_BY_TD = {{1: 0.5, 2: 1.5, 3: 3.0}}             # log-scale limits (report TDs)
N_TASKS = 8

def _make_env(L):
    return {FACTORY}(log_scale_limit=L)

# Fixed physics params per task, sampled once (seeded) so the floor measures
# only rollout noise, not a change of physics.
PARAMS = {{}}
for _td, _L in L_BY_TD.items():
    _env = _make_env(_L)
    for _i, _task in enumerate(_env.sample_tasks(N_TASKS, seed=_td)):
        PARAMS[f"td{{_td}}_t{{_i}}"] = _task
    _env.close()
TASK_DISTS = {{_td: [f"td{{_td}}_t{{_i}}" for _i in range(N_TASKS)] for _td in L_BY_TD}}
ALL_TASKS = [t for ts in TASK_DISTS.values() for t in ts]

_penv = _make_env(L_BY_TD[1]); _penv.set_task(PARAMS["td1_t0"])
_o, _ = _penv.reset()
SPEC = dict(obs_dim=int(np.ravel(_o).shape[0]),
            a_cols=int(_penv.action_space.shape[0]),
            discrete=False, n_actions=None)
_penv.close()

def _dir(reward, seed):
    return DATA / f"seed{{seed}}"

def collect(reward, seed):
    out = _dir(reward, seed); out.mkdir(parents=True, exist_ok=True)
    for _td, _L in L_BY_TD.items():
        env = _make_env(_L)
        for _i in range(N_TASKS):
            tid = f"td{{_td}}_t{{_i}}"; f = out / f"{{tid}}.npy"
            if f.exists():
                continue
            env.set_task(PARAMS[tid])
            obs, _ = env.reset(seed=1000 * seed + _i)
            rows = []
            while len(rows) < N:
                a = env.action_space.sample()
                nobs, r, term, trunc, _ = env.step(a)
                rows.append(np.concatenate([np.ravel(obs), np.ravel(a), [r], np.ravel(nobs)]))
                obs = nobs
                if term or trunc:
                    obs, _ = env.reset()
            np.save(f, np.asarray(rows[:N], dtype=np.float64))
        env.close()

def load_fn_for(reward):
    def load(task, seed):
        return np.load(_dir(reward, seed) / f"{{task}}.npy")
    return load'''

WALKER_CFG = PHYS_CFG_TMPL.format(FACTORY="Walker2dRandPhys", DATA_NAME="walkerdyn_h1")
CHEETAHDYN_CFG = PHYS_CFG_TMPL.format(FACTORY="HalfCheetahRandPhys", DATA_NAME="cheetahdyn_h1")


def md(env_title, note):
    return f"""# H1 — WTE excess for {env_title}

For each task distribution (TD): collect `(s,a,r,s')`, embed every task, and report
for **both** `(s,a,r)` and `(s,a,s')`:

- **F** — per-task noise floor (mean pairwise distance of a task's embeddings across
  seeds), averaged over the TD's tasks.
- **d(T)** — per-seed mean pairwise distance across the TD's tasks, averaged over seeds.
- **excess = d(T) − F**, for L2 and cosine.

Then the metric-property checks (non-negativity, symmetry, identity, triangle) on all
tasks pooled across TDs. Shared logic lives in `scripts/wte_h1.py`.

{note}"""


def _lines(src):
    """Notebook source is a list of lines, each keeping its trailing newline."""
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def _code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(src)}


def build(path, title, note, cfg_cell, run_collect=True):
    nb = {
        "cells": [
            _md(md(title, note)),
            _code(cfg_cell),
            _code(COLLECT_RUN),
            _code(ANALYSIS),
            _code(CLASSIFY),
        ],
        "metadata": {
            "kernelspec": {"display_name": "adv_rl", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))
    print("wrote", path.name)


# GridWorld notebook is already executed; leave it alone unless --all is passed.
import sys as _sys
_rebuild_grid = "--all" in _sys.argv
if _rebuild_grid:
    build(SCRIPTS / "h1_gridworld.ipynb", "GridWorld (6x6)",
          "Goal-varying: `(s,a,r)` should rank TD1<TD2<TD3; `(s,a,s')` is a negative "
          "control (~floor) and is identical across the two rewards.", GRID_CFG)

build(SCRIPTS / "h1_cheetahvel.ipynb", "HalfCheetahVel",
      "Goal-varying (target velocity). `h1_mujoco.HalfCheetahVelEnv` mirrors hyper "
      "(20-d obs, same reward). `(s,a,s')` is the negative control.", CHEVEL_CFG)

build(SCRIPTS / "h1_antdir.ipynb", "AntDir",
      "Goal-varying (walking direction). `h1_mujoco.AntDirEnv` mirrors hyper AntDir "
      "(obs/reward/termination; gymnasium backend, no mujoco_py). "
      "A random ant barely moves, so expect small `(s,a,r)` excess.", ANTDIR_CFG)

build(SCRIPTS / "h1_walkerdyn.ipynb", "Walker2D randomised physics",
      "Physics-varying (`L` in {0.5, 1.5, 3}): `(s,a,s')` carries the signal. "
      "Gymnasium Walker2d + RandomEnv-style param scaling (`h1_mujoco.py`).",
      WALKER_CFG)

build(SCRIPTS / "h1_cheetahdyn.ipynb", "HalfCheetah randomised physics",
      "Physics-varying (`L` in {0.5, 1.5, 3}): `(s,a,s')` carries the signal. "
      "Gymnasium HalfCheetah + RandomEnv-style param scaling (`h1_mujoco.py`). "
      "Legacy `HalfCheetahRandParams-v0` (mujoco_py) is also registered for training.",
      CHEETAHDYN_CFG)
