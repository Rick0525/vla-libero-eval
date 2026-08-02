"""Per-episode first-divergence analysis for oft_divergence_probe.py digest logs.

Usage: python oft_div_analyze.py <runA.log> <runB.log>

Reports: episode-outcome flips between the two runs, how many episodes show
any step-level divergence, and which digest field(s) differ at each episode's
first divergent step (sim = physics, st = policy state input, img = agentview
render, wri = wrist render, act = executed action).

Finding on our stack (2026-08-02, task 7, 2 processes x 50 episodes): all 50
episodes diverge; the first divergence involves a render field every time
(wrist 45/50, agentview 9/50), never sim/st/act alone -- localizing the
closed-loop nondeterminism to off-screen EGL rendering.
"""
import re
import sys
from collections import Counter


def load(path):
    rows, eps = {}, {}
    for line in open(path):
        m = re.match(r"^(\d+) (\d+) sim=(\S+) st=(\S+) img=(\S+) wri=(\S+) act=(\S+)", line)
        if m:
            ep, t = int(m.group(1)), int(m.group(2))
            rows[(ep, t)] = dict(sim=m.group(3), st=m.group(4), img=m.group(5), wri=m.group(6), act=m.group(7))
        m2 = re.match(r"^EP (\d+) success=(\w+)", line)
        if m2:
            eps[int(m2.group(1))] = m2.group(2)
    return rows, eps


A, epA = load(sys.argv[1])
B, epB = load(sys.argv[2])

flips = [e for e in epA if epA.get(e) != epB.get(e)]
n_a = sum(v == "True" for v in epA.values())
n_b = sum(v == "True" for v in epB.values())
print("outcome flips A vs B:", flips if flips else "none", f"(A successes={n_a}, B={n_b})")

firstdiff = {}
for ep in sorted(epA):
    for t in sorted(t for (e, t) in A if e == ep and (e, t) in B):
        d = [k for k in ("sim", "st", "img", "wri", "act") if A[(ep, t)][k] != B[(ep, t)][k]]
        if d:
            firstdiff[ep] = (t, tuple(d))
            break

print("episodes with any divergence:", len(firstdiff), "/", len(epA))
print("first-diff field pattern counts:", dict(Counter(v[1] for v in firstdiff.values())))
print("per-episode first-diff (ep: step, fields):", firstdiff)
