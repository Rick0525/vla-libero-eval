#!/usr/bin/env bash
# Exp E launcher (repair retrain, two sequential DDP arms on GPU0+1).
# Usage: launch_exp_e.sh probe|ctrl|boost|arms
#   probe — 300-step full-pipeline check on the boost dataset (house rule:
#           probe before every real launch; verify b64 smpl counter in log)
#   ctrl  — control arm: baseline dataset, exact b64 recipe (recipe gate)
#   boost — boost arm: local/libero_task5boost_k4, same recipe
#   arms  — ctrl then boost, chained (overnight mode)
set -euo pipefail
ARM="${1:?usage: launch_exp_e.sh probe|ctrl|boost|arms}"

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
export CUDA_VISIBLE_DEVICES=0,1

case "$ARM" in
  probe)
    export RUN_TAG=exp_e_probe DATASET_REPO=local/libero_task5boost_k4
    export TRAIN_STEPS=300 SAVE_FREQ=300 LOG_FREQ=50
    rm -rf "${VLA_TRAIN_OUTPUT_DIR}/smolvla_spatial_${RUN_TAG}"  # probes are disposable
    bash scripts/train_smolvla_spatial_b64.sh
    ;;
  ctrl)
    export RUN_TAG=exp_e_ctrl_20260728
    bash scripts/train_smolvla_spatial_b64.sh
    ;;
  boost)
    export RUN_TAG=exp_e_boost_k4_20260728 DATASET_REPO=local/libero_task5boost_k4
    bash scripts/train_smolvla_spatial_b64.sh
    ;;
  arms)
    "$0" ctrl 2>&1 | tee "$HOME/vla_lab/logs/train_exp_e_ctrl_20260728.log"
    "$0" boost 2>&1 | tee "$HOME/vla_lab/logs/train_exp_e_boost_20260728.log"
    ;;
esac
