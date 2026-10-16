#!/usr/bin/env bash
# Train RNN+HN ablations on HalfCheetahVel with a shared task distribution.
#
# TaskDist choice: cheetah_vel_taskdist3 = Uniform[0, 3]
#   - Same range as original HalfCheetahVel-v0 / paper-style CheetahVel
#   - Highest within-dist WTE among TD1/2/3 → enough task diversity that
#     full HN vs head-only vs FiLM can diverge
#   - TD1 [2.5,3] / TD2 [0.5,1] are narrower; better as follow-ups
#
# Usage (from anywhere):
#   bash bash_scripts/run_chevel.sh
#   NUM_FRAMES=5000000 SEED=73 bash bash_scripts/run_chevel.sh
#   RUNS="full head" bash bash_scripts/run_chevel.sh   # subset
#
# Paths: this file lives in <repo>/bash_scripts/ → REPO_ROOT=.. → runs in <repo>/hyper/
# Requires: conda env with mujoco/gymnasium (e.g. adv_rl), GPU recommended.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYPER_DIR="${REPO_ROOT}/hyper"

if [[ ! -f "${HYPER_DIR}/main.py" ]]; then
  echo "ERROR: expected ${HYPER_DIR}/main.py (REPO_ROOT=${REPO_ROOT})" >&2
  exit 1
fi

ENV_TYPE="${ENV_TYPE:-cheetah_vel_taskdist3}"
NUM_FRAMES="${NUM_FRAMES:-75000000}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
SEED="${SEED:-73}"
LR_POLICY="${LR_POLICY:-0.0003}"
RESULTS_LOG_DIR="${RESULTS_LOG_DIR:-}"   # empty → hyper/logs
RUNS="${RUNS:-full head film}"           # space-separated: full | head | film

cd "${HYPER_DIR}"
echo "cwd: $(pwd)"
echo "env-type: ${ENV_TYPE}  frames: ${NUM_FRAMES}  procs: ${NUM_PROCESSES}  seed: ${SEED}"
echo "runs: ${RUNS}"

COMMON=(
  --env-type "${ENV_TYPE}"
  --use_hypernet True
  --hypernet_input latent
  --policy_layers 32 32
  --policy_state_embedding_dim 25
  --policy_latent_embedding_dim 25
  --encoder_gru_hidden_size 256
  --init_hyper_for_policy True
  --init_adjust_weights False
  --init_adjust_bias True
  --hypernet_head_bias True
  --disable_decoder True
  --disable_stochasticity_in_latent True
  --rlloss_through_encoder True
  --norm_latent_for_policy False
  --precollect_len 0
  --pretrain_len 0
  --pass_state_to_policy True
  --pass_latent_to_policy True
  --pass_task_to_policy False
  --latent_dim 5
  --num_processes "${NUM_PROCESSES}"
  --policy_num_steps 400
  --num_frames "${NUM_FRAMES}"
  --lr_policy "${LR_POLICY}"
  --seed "${SEED}"
)

if [[ -n "${RESULTS_LOG_DIR}" ]]; then
  COMMON+=(--results_log_dir "${RESULTS_LOG_DIR}")
fi

run_one() {
  local name="$1"
  shift
  echo "======== starting: ${name} ========"
  python main.py \
    --exp_label "chevel_td3_rnn_${name}" \
    "${COMMON[@]}" \
    "$@"
  echo "======== finished: ${name} ========"
}

for run in ${RUNS}; do
  case "${run}" in
    full)
      # Original RNN + full HN (all layers generated)
      run_one full \
        --use_film False \
        --policy_layers_hyper 1 1 1
      ;;
    head)
      # Head-only HN
      run_one head \
        --use_film False \
        --policy_layers_hyper 0 0 1
      ;;
    film)
      # FiLM: shared base W, HN predicts scale+bias (init_b, same as RNN+HN)
      run_one film \
        --use_film True \
        --policy_layers_hyper 1 1 1
      ;;
    *)
      echo "Unknown run '${run}' (expected: full head film)" >&2
      exit 1
      ;;
  esac
done

echo "All requested runs done. Logs under ${RESULTS_LOG_DIR:-${HYPER_DIR}/logs}"
