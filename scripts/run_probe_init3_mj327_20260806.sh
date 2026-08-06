#!/usr/bin/env bash
# init3 poison-layout re-probe under mujoco 3.2.7 (Rick approved 2026-08-06):
# decide whether the W3 "poison layout" (SmolVLA 0/13 hard zero, pi0.5 5/10,
# both measured under 3.8.1 = broken-premise physics) survives healthy physics.
# Protocol verbatim from Exp F / 7-31 pi0.5 probe: task5, init-index 3 pinned,
# 10 rollouts, policy seeds 2000..2009. GPU2 only. Restores mujoco 3.8.1 on exit.
set -uo pipefail

PY=$HOME/vla_lab/.venv/bin/python
UV=$HOME/.local/bin/uv

restore() {
  echo "[restore] reinstalling mujoco 3.8.1"
  $UV pip install --python "$PY" --offline mujoco==3.8.1 \
    || $UV pip install --python "$PY" mujoco==3.8.1 \
    || { echo "RESTORE_FAILED mujoco=$($PY -c "import mujoco;print(mujoco.__version__)")"; return; }
  echo "[restore] venv back to mujoco $($PY -c "import mujoco;print(mujoco.__version__)")"
}
trap restore EXIT

echo "[downgrade] mujoco -> 3.2.7"
$UV pip install --python "$PY" --offline mujoco==3.2.7 \
  || $UV pip install --python "$PY" mujoco==3.2.7 \
  || { echo "DOWNGRADE_FAILED"; exit 1; }

source "$HOME/vla_lab/scripts/env.sh"
source /mnt/hdd16t/rick/vla_lab/vla-eval-harness/scripts/env.local.sh
export CUDA_VISIBLE_DEVICES=2
export MUJOCO_GL=egl

MJV=$("$PY" -c "import mujoco; print(mujoco.__version__)")
echo "[preflight] mujoco=$MJV gpu=$CUDA_VISIBLE_DEVICES"
[ "$MJV" = "3.2.7" ] || { echo "PREFLIGHT_FAIL mujoco=$MJV"; exit 1; }

cd /mnt/hdd16t/rick/vla_lab/vla-eval-harness

echo "[stage 1/2] SmolVLA 30k init3 x10 @mj327 $(date +%F_%T)"
python scripts/attribution_probe.py --mode rollout --task-id 5 \
  --init-index 3 --n-rollouts 10 --base-seed 2000 \
  --out-dir "$VLA_EVAL_OUTPUT_DIR/attr_f_task5_init3_mj327_20260806"
rc=$?
if [ "$rc" -eq 0 ]; then echo "SMOLVLA_INIT3_MJ327_DONE_OK"; else echo "SMOLVLA_INIT3_MJ327_DONE_FAIL rc=$rc"; fi

echo "[stage 2/2] pi0.5 init3 x10 @mj327 $(date +%F_%T)"
python scripts/attribution_probe.py --mode rollout --task-id 5 \
  --init-index 3 --n-rollouts 10 --base-seed 2000 \
  --model-path /mnt/hdd16t/rick/vla_lab/models/pi05-libero \
  --n-action-steps 10 \
  --out-dir "$VLA_EVAL_OUTPUT_DIR/attr_f_pi05_task5_init3_mj327_20260806"
rc2=$?
if [ "$rc2" -eq 0 ]; then echo "PI05_INIT3_MJ327_DONE_OK"; else echo "PI05_INIT3_MJ327_DONE_FAIL rc=$rc2"; fi

echo "ALL_DONE smolvla_rc=$rc pi05_rc=$rc2 $(date +%F_%T)"
