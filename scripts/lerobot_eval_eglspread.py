#!/usr/bin/env python
# EGL-spread diagnostic wrapper around lerobot-eval.
#
# Motivation: with --eval.batch_size=10 the LIBERO async vector env spawns 10
# worker processes, and robosuite's EGL display selection (egl_context.py)
# falls back to CUDA_VISIBLE_DEVICES when MUJOCO_EGL_DEVICE_ID is unset — so
# every worker renders on the same GPU that also runs policy inference. The
# measured async speedup over serial eval was only ~1.3-2x; the suspected
# culprit is those 10 EGL contexts time-slicing one GPU. This wrapper spreads
# worker rendering across GPUs to test that hypothesis.
#
# Usage: identical CLI to lerobot-eval, plus one env knob:
#   EVAL_EGL_DEVICES="1,2"  -> worker i renders on EGL device (1,2)[i % 2]
# Device ids index the machine's physical EGL device list (all GPUs), not the
# parent's CUDA_VISIBLE_DEVICES. Unset/empty -> behaves exactly as lerobot-eval.
#
# Why this works: robosuite reads MUJOCO_EGL_DEVICE_ID when it creates the EGL
# display (call time, inside the worker), so setting the env var in the worker
# before the wrapped factory runs is sufficient. Each factory callable below
# executes inside its own forkserver worker process, never in the parent.
import os

_DEVICES = [d.strip() for d in os.environ.get("EVAL_EGL_DEVICES", "").split(",") if d.strip()]

if _DEVICES:
    import gymnasium as gym

    import lerobot.envs.utils as _lu

    # AsyncVectorEnv probes env_fns[0]() in the PARENT to learn the spaces, so
    # the factory below also runs there once. Only workers may get their env
    # mutated: changing CUDA_VISIBLE_DEVICES in the parent after torch has
    # initialized CUDA breaks dynamo's device enumeration.
    _PARENT_PID = os.getpid()

    def _with_egl_device(device: str, fn):
        def _factory():
            if os.getpid() != _PARENT_PID:
                # Inside a worker. Widen CUDA_VISIBLE_DEVICES so EGL device
                # enumeration cannot be filtered down to the parent's
                # inference GPU; workers do CPU physics + EGL rendering only,
                # no CUDA compute, so this moves no policy computation.
                visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dict.fromkeys(visible + _DEVICES))
                os.environ["MUJOCO_EGL_DEVICE_ID"] = device
            return fn()

        return _factory

    def _ensure(self):
        # Mirrors _LazyAsyncVectorEnv._ensure, with round-robin EGL devices.
        if self._env is None:
            fns = [
                _with_egl_device(_DEVICES[i % len(_DEVICES)], fn)
                for i, fn in enumerate(self._env_fns)
            ]
            self._env = gym.vector.AsyncVectorEnv(fns, context="forkserver", shared_memory=True)

    _lu._LazyAsyncVectorEnv._ensure = _ensure
    print(f"[eglspread] worker EGL devices: {_DEVICES} (round-robin)")

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    main()
