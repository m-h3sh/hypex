#!/usr/bin/env bash
# Run RNN+HN on 6x6 sparse TaskDist 1–3 (random start).
# For each task dist: full HN policy params, then head-only.
#
# Full:  --policy_layers_hyper 1 1 1  (both hidden layers + heads)
# Head:  --policy_layers_hyper 0 0 1  (shared 32,32; HN makes actor/critic heads)
#
# Env: GridNavi-TaskDist{1,2,3}-v0 — 6x6, sparse (+1 / -0.1), random start.
# Logs: hyper/logs_sparse/logs_GridNavi-TaskDist{N}-v0/
#
# Budget: 16000 PPO updates each.
#   frames = updates * policy_num_steps * num_processes
#          = 16000 * 60 * 16 = 15_360_000
set -euo pipefail

cd "$(dirname "$0")/../hyper"

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
  --log_hn_weight_div False
  --policy_num_steps "${POLICY_NUM_STEPS}"
  --num_processes "${NUM_PROCESSES}"
  --num_frames "${NUM_FRAMES}"
  --seed "${SEED}"
  --results_log_dir "${LOG_DIR}"
)

echo "Env: 6x6 sparse TaskDist, random start"
echo "Order per dist: full HN params, then head-only"
echo "Shared budget: ${NUM_UPDATES} updates  (${NUM_FRAMES} frames)"
echo "Seed: ${SEED}"
echo "Logs: ${LOG_DIR}/"
echo

for DIST in 1 2 3; do
  echo "========== TaskDist ${DIST} — full HN =========="
  python main.py \
    --env-type "gridworld_taskdist${DIST}" \
    --exp_label "td${DIST}_6_full" \
    --policy_layers_hyper 1 1 1 \
    "${COMMON[@]}"
  echo

  echo "========== TaskDist ${DIST} — head only =========="
  python main.py \
    --env-type "gridworld_taskdist${DIST}" \
    --exp_label "td${DIST}_6_sparse_head" \
    --policy_layers_hyper 0 0 1 \
    "${COMMON[@]}"
  echo
done

echo "Done. Logs under:"
echo "  ${LOG_DIR}/logs_GridNavi-TaskDist{1,2,3}-v0/"
echo "Labels: td{1,2,3}_6_sparse_{full,head}"
