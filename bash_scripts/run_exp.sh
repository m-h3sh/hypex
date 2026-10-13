#!/usr/bin/env bash
# Remaining jobs after Orig 6x6 finished: TaskDist 8 only.
# 6x6 sparse, H=20, random start, all 36 goals. Weight-div logging on.
#
# Full:  --policy_layers_hyper 1 1 1
# Head:  --policy_layers_hyper 0 0 1
#
# Budget: 16000 PPO updates each.
#   frames = 16000 * 60 * 16 = 15_360_000
set -euo pipefail

cd "$(dirname "$0")/../hyper"

# Cache sudo now so end-of-run pmset/sleep works without a prompt hours later.
sudo -v
# Refresh sudo timestamp while this script is running.
(
  while true; do
    sudo -n true
    sleep 60
    kill -0 "$$" || exit
  done
) 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true' EXIT

SEED=73
NUM_UPDATES="${NUM_UPDATES:-16000}"
POLICY_NUM_STEPS=60
NUM_PROCESSES=16
NUM_FRAMES=$((NUM_UPDATES * POLICY_NUM_STEPS * NUM_PROCESSES))
LOG_DIR="logs_sparse"

COMMON=(
  --use_hypernet True
  --hypernet_input latent
  --policy_layers 32 32
  --policy_state_embedding_dim 25
  --policy_latent_embedding_dim 25
  --init_hyper_for_policy True
  --init_adjust_weights False
  --init_adjust_bias True
  --hypernet_head_bias True
  --disable_decoder True
  --disable_stochasticity_in_latent True
  --rlloss_through_encoder True
  --norm_latent_for_policy False
  --pass_state_to_policy True
  --pass_latent_to_policy True
  --pass_task_to_policy False
  --pass_belief_to_policy False
  --latent_dim 5
  --action_embedding_size 0
  --log_hn_weight_div True
  --policy_num_steps "${POLICY_NUM_STEPS}"
  --num_processes "${NUM_PROCESSES}"
  --num_frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --results_log_dir "${LOG_DIR}"
)

echo "Remaining: TaskDist 8 only (Orig already done)"
echo "6x6 sparse, H=20, random start, 36 goals, seed ${SEED}"
echo "Order: full HN, then head-only"
echo "Budget: ${NUM_UPDATES} updates  (${NUM_FRAMES} frames)"
echo "Logs: ${LOG_DIR}/"
echo

echo "========== TaskDist 8 — full HN =========="
python main.py \
  --env-type "gridworld_taskdist8" \
  --exp_label "td_6_fullgoal_full" \
  --policy_layers_hyper 1 1 1 \
  "${COMMON[@]}"
echo

echo "========== TaskDist 8 — head only =========="
python main.py \
  --env-type "gridworld_taskdist8" \
  --exp_label "td_6_fullgoal_head" \
  --policy_layers_hyper 0 0 1 \
  "${COMMON[@]}"
echo

echo "Done. Logs under:"
echo "  ${LOG_DIR}/logs_GridNavi-TaskDist8-v0/  labels td_6_fullgoal_{full,head}"

echo "Re-enabling sleep (disablesleep 0), then sleeping the Mac..."
sudo pmset -a disablesleep 0
sudo pmset sleepnow
