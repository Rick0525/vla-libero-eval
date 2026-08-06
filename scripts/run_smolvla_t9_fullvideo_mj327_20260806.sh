#!/usr/bin/env bash
# SmolVLA 30k task9 full-video rerun @mujoco 3.2.7 (Rick approved 2026-08-06):
# t9 failures (41/50 in both v1/v2 runs) were never recorded -- stock eval caps
# video at 10 eps/task. Rerun task 9 alone with EVAL_RENDER_ALL_N so Rick can
# adjudicate whether 0.45B commits stove/cabinet referent confusion (currently
# only verified absent on t7). Same scale as v2 main table (3.2.7); GPU2;
# restores mujoco 3.8.1 on exit.
set -uo pipefail

PY=$HOME/vla_lab/.venv/bin/python
UV=$HOME/.local/bin/uv

restore() {
  echo "[restore] reinstalling mujoco 3.8.1"
  $UV pip install --python "$PY" --offline mujoco==3.8.1 \
    || $UV pip install --python "$PY" mujoco==3.8.1 \
    || { echo "RESTORE_FAILED"; return; }
  echo "[restore] venv back to mujoco $($PY -c "import mujoco;print(mujoco.__version__)")"
}
trap restore EXIT

echo "[downgrade] mujoco -> 3.2.7"
$UV pip install --python "$PY" --offline mujoco==3.2.7 \
  || $UV pip install --python "$PY" mujoco==3.2.7 || { echo "DOWNGRADE_FAILED"; exit 1; }

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl

MJV=$("$PY" -c "import mujoco; print(mujoco.__version__)")
echo "[preflight] mujoco=$MJV gpu=$CUDA_VISIBLE_DEVICES"
[ "$MJV" = "3.2.7" ] || { echo "PREFLIGHT_FAIL mujoco=$MJV"; exit 1; }

cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness

TRAIN_RUN=smolvla_spatial_b64_30k CKPT_STEP=030000 \
  EVAL_N_EPISODES=50 EVAL_TASK_IDS="[9]" \
  EVAL_ENTRY="python scripts/lerobot_eval_fullvideo.py" EVAL_RENDER_ALL_N=10000 \
  EVAL_TAG=t9full_mj327_20260806 bash scripts/eval_checkpoint_spatial.sh
rc=$?
if [ "$rc" -eq 0 ]; then echo "T9_FULLVIDEO_DONE_OK"; else echo "T9_FULLVIDEO_DONE_FAIL rc=$rc"; fi
echo "ALL_DONE rc=$rc $(date +%F_%T)"
