#!/usr/bin/env bash
# Train RNN+HN ablations (full / head / FiLM) on GridNavi TaskDist 1–4.
#
# Same conditioning ablations as bash_scripts/run_chevel.sh, but gridworld
# hyperparameters (not Cheetah XL sizes).
#
# Envs (registered GridNavi-Dense6-TaskDist{k}-v0); order TD4 → TD3 → TD2 → TD1.
# These exactly match the MDP used to compute the WTE numbers in
# scripts/exp2_copy.ipynb (dense twin of the sparse probe in exp2.ipynb):
#   6x6, dense Manhattan reward (-distance/6; +1 at goal), H=20,
#   uniform random start; TD1–3 have 4 goals each, TD4 has all 36 cells
#   TD4: every (x, y) in 6x6               [all-cells / max diversity]
#   TD3: (0,0), (0,5), (5,0), (5,5)       [four corners]
#   TD2: (3,3), (5,3), (3,5), (5,5)       [mid/edge spread]
#   TD1: (5,5), (5,4), (4,5), (4,4)       [top-right cluster]
# (The old sparse 6x6 logs in logs_gridworld/ are untouched.)
#
# Hyperparameters match all_envs.py "grid" shared_arguments + RNN_HN (init_b),
# with policy_state_embedding_dim 25 (reported best for Grid RNN+HN):
#   - policy_layers 32 32
#   - policy_state/latent_embedding_dim 25
#   - encoder_gru_hidden_size 256
#   - num_frames 2e6
#   - policy_num_steps 60, num_processes 16
#   - max_rollouts_per_task 4 (grid BAMDP default)
#   - action_embedding_size 0 (discrete)
#   - lr_policy 7e-4, lr_vae 1e-3
#   - RNN+HN: disable_decoder, non-var latent, rlloss_through_encoder, init_b
#
# Ablations (same as run_chevel.sh):
#   full  = RNN+HN          policy_layers_hyper 1 1 1
#   head  = head-only HN    policy_layers_hyper 0 0 1
#   film  = FiLM_init_b     use_film + 1 1 1
#
# Usage:
#   bash bash_scripts/run_exp.sh
#   DISTS="1 3" RUNS="full film" bash bash_scripts/run_exp.sh
#   NUM_FRAMES=1000000 bash bash_scripts/run_exp.sh
#
# Logs: <repo>/logs_gridworld_dense6/logs_GridNavi-Dense6-TaskDist{k}-v0/
# Labels: grid_dense6_td{k}_rnn_{full,head,film}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYPER_DIR="${REPO_ROOT}/hyper"

if [[ ! -f "${HYPER_DIR}/main.py" ]]; then
  echo "ERROR: expected ${HYPER_DIR}/main.py (REPO_ROOT=${REPO_ROOT})" >&2
  exit 1
fi

# --- env / budget (override via env vars) ---
DISTS="${DISTS:-4 3 2 1}"
NUM_FRAMES="${NUM_FRAMES:-2000000}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
POLICY_NUM_STEPS="${POLICY_NUM_STEPS:-60}"
SEED="${SEED:-73}"
LR_POLICY="${LR_POLICY:-0.0007}"
LR_VAE="${LR_VAE:-0.001}"
RESULTS_LOG_DIR="${RESULTS_LOG_DIR:-${REPO_ROOT}/logs_gridworld_dense6}"
RUNS="${RUNS:-full head film}"

cd "${HYPER_DIR}"
echo "cwd: $(pwd)"
echo "dists: ${DISTS}  frames: ${NUM_FRAMES}  procs: ${NUM_PROCESSES}  steps/update: ${POLICY_NUM_STEPS}"
echo "lr_policy: ${LR_POLICY}  lr_vae: ${LR_VAE}  seed: ${SEED}"
echo "runs: ${RUNS}"
echo "logs: ${RESULTS_LOG_DIR}"

# Shared RNN+HN / grid recipe (env-type + env_name set per dist below).
COMMON=(
  # --- RNN+HN / RL2-style (non-variational end-to-end) ---
  --use_hypernet True
  --hypernet_input latent
  --disable_decoder True
  --disable_stochasticity_in_latent True
  --sample_embeddings False
  --rlloss_through_encoder True
  --norm_latent_for_policy False
  --precollect_len 0
  --pretrain_len 0
  --init_hyper_for_policy True
  --init_adjust_weights False
  --init_adjust_bias True
  --hypernet_head_bias True
  --log_hn_weight_div True
  # --- sizes (grid shared + state emb 25) ---
  --policy_layers 32 32
  --policy_state_embedding_dim 25
  --policy_latent_embedding_dim 25
  --policy_task_embedding_dim 25
  --encoder_gru_hidden_size 256
  # --- policy I/O ---
  --pass_state_to_policy True
  --pass_latent_to_policy True
  --pass_task_to_policy False
  --pass_belief_to_policy False
  --norm_state_for_policy True
  --norm_rew_for_policy True
  --latent_dim 5
  --action_embedding_size 0
  --state_embedding_size 32
  --reward_embedding_size 8
  # --- PPO / rollout (grid varibad defaults) ---
  --policy_num_steps "${POLICY_NUM_STEPS}"
  --num_processes "${NUM_PROCESSES}"
  --num_frames "${NUM_FRAMES}"
  --lr_policy "${LR_POLICY}"
  --lr_vae "${LR_VAE}"
  --ppo_num_epochs 2
  --ppo_num_minibatch 4
  --policy_gamma 0.95
  --policy_tau 0.95
  --policy_entropy_coef 0.1
  --max_rollouts_per_task 4
  --tbptt_stepsize None
  --eval_interval 5
  --seed "${SEED}"
  --results_log_dir "${RESULTS_LOG_DIR}"
)

run_one() {
  local dist_id="$1"
  local name="$2"
  shift 2
  local env_type="gridworld_dense6_taskdist${dist_id}"
  local env_name="GridNavi-Dense6-TaskDist${dist_id}-v0"
  echo "======== TD${dist_id} / ${name}  (${env_name}) ========"
  python main.py \
    --env-type "${env_type}" \
    --env_name "${env_name}" \
    --exp_label "grid_dense6_td${dist_id}_rnn_${name}" \
    "${COMMON[@]}" \
    "$@"
  echo "======== finished: TD${dist_id} / ${name} ========"
}

for dist_id in ${DISTS}; do
  case "${dist_id}" in
    1|2|3|4) ;;
    *)
      echo "Unknown dist '${dist_id}' (expected: 1 2 3 4)" >&2
      exit 1
      ;;
  esac
  for run in ${RUNS}; do
    case "${run}" in
      full)
        run_one "${dist_id}" full \
          --use_film False \
          --policy_layers_hyper 1 1 1
        ;;
      head)
        run_one "${dist_id}" head \
          --use_film False \
          --policy_layers_hyper 0 0 1
        ;;
      film)
        run_one "${dist_id}" film \
          --use_film True \
          --policy_layers_hyper 1 1 1
        ;;
      *)
        echo "Unknown run '${run}' (expected: full head film)" >&2
        exit 1
        ;;
    esac
  done
done

echo "All requested runs done. Logs under ${RESULTS_LOG_DIR}/logs_GridNavi-Dense6-TaskDist{1,2,3,4}-v0/"
