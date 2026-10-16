#!/usr/bin/env bash
# Train RNN+HN ablations on HalfCheetahVel with a shared task distribution.
#
# TaskDist: cheetah_vel_taskdist3 = Uniform[0, 3]
#   Same velocity range as src/hyper HalfCheetahVelEnv.sample_tasks (and HalfCheetahVel-v0).
#
# Hyperparameters match the *effective* original chevel + RNN_HN launch from
# hyper/experiment_sets (all_envs.py "chevel" shared_arguments merged with RNN_HN),
# where experiment/model args override shared:
#   - policy_layers 256 128          (make_pol_size XL; overrides shared "32 32")
#   - policy_state_embedding_dim 256 (XL)
#   - policy_latent_embedding_dim 25 (shared)
#   - encoder_gru_hidden_size 256    (shared)
#   - num_frames 75e6                (shared)
#   - tbptt_stepsize None            (shared)
#   - num_processes 16               (args_cheetah_vel_varibad default)
#   - lr_policy 7e-4                 (varibad config default; paper also sweeps)
#   - RNN+HN: disable_decoder, rlloss_through_encoder, non-var latent, init_b, …
#
# Ablations (same backbone / env):
#   full  = RNN+HN          policy_layers_hyper 1 1 1
#   head  = head-only HN    policy_layers_hyper 0 0 1
#   film  = FiLM_init_b     use_film + 1 1 1
#
# Usage:
#   bash bash_scripts/run_chevel.sh
#   NUM_FRAMES=2000000 NUM_PROCESSES=8 bash bash_scripts/run_chevel.sh
#   RUNS="full head" bash bash_scripts/run_chevel.sh
#
# Paths: <repo>/bash_scripts/ → REPO_ROOT=.. → runs <repo>/hyper/main.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYPER_DIR="${REPO_ROOT}/hyper"

if [[ ! -f "${HYPER_DIR}/main.py" ]]; then
  echo "ERROR: expected ${HYPER_DIR}/main.py (REPO_ROOT=${REPO_ROOT})" >&2
  exit 1
fi

# --- env / budget (override via env vars) ---
ENV_TYPE="${ENV_TYPE:-cheetah_vel_taskdist3}"
NUM_FRAMES="${NUM_FRAMES:-75000000}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
SEED="${SEED:-73}"
LR_POLICY="${LR_POLICY:-0.0007}"
LR_VAE="${LR_VAE:-0.001}"
RESULTS_LOG_DIR="${RESULTS_LOG_DIR:-}"
RUNS="${RUNS:-full head film}"

cd "${HYPER_DIR}"
echo "cwd: $(pwd)"
echo "env-type: ${ENV_TYPE}  frames: ${NUM_FRAMES}  procs: ${NUM_PROCESSES}"
echo "lr_policy: ${LR_POLICY}  lr_vae: ${LR_VAE}  seed: ${SEED}"
echo "runs: ${RUNS}"

# Shared with original chevel + RNN_HN (see experiment_sets merge above).
# Remaining PPO / encoder / VAE knobs come from args_cheetah_vel_varibad defaults
# (policy_num_steps 400, latent_dim 5, action/state/reward emb 16/32/16, etc.).
COMMON=(
  --env-type "${ENV_TYPE}"
  --env_name HalfCheetahVel-TaskDist3-v0
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
  # --- sizes (XL + chevel shared latent/gru) ---
  --policy_layers 256 128
  --policy_state_embedding_dim 256
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
  --action_embedding_size 16
  --state_embedding_size 32
  --reward_embedding_size 16
  # --- PPO / rollout (varibad CheetahVel defaults) ---
  --policy_num_steps 400
  --num_processes "${NUM_PROCESSES}"
  --num_frames "${NUM_FRAMES}"
  --lr_policy "${LR_POLICY}"
  --lr_vae "${LR_VAE}"
  --ppo_num_epochs 2
  --ppo_num_minibatch 4
  --policy_gamma 0.97
  --policy_tau 0.9
  --policy_entropy_coef 0.01
  --max_rollouts_per_task 1
  --tbptt_stepsize None
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
      # Original RNN+HN: full weight generation
      run_one full \
        --use_film False \
        --policy_layers_hyper 1 1 1
      ;;
    head)
      # Head-only HN (same backbone sizes)
      run_one head \
        --use_film False \
        --policy_layers_hyper 0 0 1
      ;;
    film)
      # FiLM_Bias-HyperInit (fair vs RNN+HN init_b)
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
