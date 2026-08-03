"""RNG consumer tracer v2 (2026-08-03, model-free).

v1 found ep0 dummy steps consume nothing, yet precheck2's 14-episode context
loop advanced the global np.random stream. So the consumer fires only in some
episodes/segments. This version replicates precheck2's context loop exactly
(eps 0..13: reset + set_init_state + 60 dummy steps) and accounts per segment
for both np.random and python-random global streams, with np.random public
functions wrapped to record caller sites.
"""
import collections
import random as pyrandom
import traceback

import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(7)
inits = suite.get_task_init_states(7)
env, _ = get_libero_env(task, "openvla", resolution=256)

nph = lambda: hash(np.random.get_state()[1].tobytes())
pyh = lambda: hash(str(pyrandom.getstate()))

sites = collections.Counter()
def wrap(name):
    orig = getattr(np.random, name)
    def w(*args, **kw):
        st = traceback.extract_stack(limit=8)[:-1]
        key = " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in st[-4:])
        sites[f"np.random.{name} | {key}"] += 1
        return orig(*args, **kw)
    setattr(np.random, name, w)

for n in ["random", "random_sample", "uniform", "randn", "normal", "randint",
          "choice", "rand", "permutation", "shuffle"]:
    try:
        wrap(n)
    except AttributeError:
        pass

for ep in range(14):
    a_np, a_py = nph(), pyh()
    env.reset()
    b_np, b_py = nph(), pyh()
    env.set_init_state(inits[ep])
    c_np, c_py = nph(), pyh()
    nsteps = 0
    for t in range(60):
        d = nph()
        env.step(get_libero_dummy_action("openvla"))
        if nph() != d:
            nsteps += 1
    print(f"ep{ep:02d} reset:np={'X' if b_np != a_np else '.'} py={'X' if b_py != a_py else '.'}"
          f"  set_init:np={'X' if c_np != b_np else '.'} py={'X' if c_py != b_py else '.'}"
          f"  steps_np={nsteps}/60", flush=True)

print("top np.random call sites:")
for k, v in sites.most_common(12):
    print(v, k)
print("TRACER2_DONE")
