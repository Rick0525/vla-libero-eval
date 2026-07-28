#!/usr/bin/env bash
# Exp E curve daemons: score each arm's 5000-step checkpoint grid
# (curve protocol: n_action_steps=10, 5 eps/task) via eval_daemon_spatial.sh.
# Usage: launch_exp_e_curves.sh <train_run> <gpu>
set -uo pipefail
ARM="$1"; GPU="$2"
source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
TRAIN_RUN="$ARM" TRAIN_TMUX=exp_e_arms BLOCK_TMUX=exp_e_noblock EVAL_GPU="$GPU" \
  bash scripts/eval_daemon_spatial.sh
