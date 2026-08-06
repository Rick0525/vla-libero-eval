#!/usr/bin/env bash
# Leaderboard mujoco-version rerun (Rick approved 2026-08-04 night):
# SmolVLA 30k + pi0.5 re-evaluated under mujoco 3.2.7 (the OFT official-env
# version) to remove the cross-stack physics confound found on 8/04.
# Commands/config/seed are verbatim copies of the 7/30 leaderboard runs
# (run_pi05_official_20260730.sh / run_eval_s500_20260730.sh); only the GPU
# (2 -- GPU0/1 belong to the RL project tonight) and output dirs differ.
# The venv is temporarily at mujoco 3.2.7; this script restores 3.8.1 on exit.
set -uo pipefail

PY=$HOME/vla_lab/.venv/bin/python
UV=$HOME/.local/bin/uv

restore() {
  echo "[restore] reinstalling mujoco 3.8.1"
  $UV pip install --python "$PY" --offline mujoco==3.8.1 \
    || $UV pip install --python "$PY" mujoco==3.8.1 \
    || { echo "RESTORE_FAILED mujoco=$($PY -c 'import mujoco;print(mujoco.__version__)')"; return; }
  echo "[restore] venv back to mujoco $($PY -c 'import mujoco;print(mujoco.__version__)')"
}
trap restore EXIT

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl

MJV=$("$PY" -c 'import mujoco; print(mujoco.__version__)')
echo "[preflight] mujoco=$MJV gpu=$CUDA_VISIBLE_DEVICES"
[ "$MJV" = "3.2.7" ] || { echo "PREFLIGHT_FAIL mujoco=$MJV"; exit 1; }

echo "[stage 1/2] pi0.5 official500 @mj327 $(date +%F_%T)"
lerobot-eval \
  --policy.path=/mnt/hdd16t/rick/vla_lab/models/pi05-libero \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --seed=1000 \
  --output_dir=/mnt/hdd16t/rick/vla_lab/eval_runs/pi05_libero_official500_mj327_20260804
rc=$?
if [ "$rc" -eq 0 ]; then echo "PI05_MJ327_DONE_OK"; else echo "PI05_MJ327_DONE_FAIL rc=$rc"; fi

echo "[stage 2/2] SmolVLA 30k official500 @mj327 $(date +%F_%T)"
cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness
TRAIN_RUN=smolvla_spatial_b64_30k CKPT_STEP=030000 EVAL_N_EPISODES=50 \
  EVAL_TAG=official500_mj327_20260804 bash scripts/eval_checkpoint_spatial.sh
rc2=$?
if [ "$rc2" -eq 0 ]; then echo "SMOLVLA_MJ327_DONE_OK"; else echo "SMOLVLA_MJ327_DONE_FAIL rc=$rc2"; fi

echo "ALL_DONE pi05_rc=$rc smolvla_rc=$rc2 $(date +%F_%T)"
