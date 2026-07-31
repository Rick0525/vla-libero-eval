#!/usr/bin/env python
# Render-all wrapper around lerobot-eval.
#
# Stock lerobot_eval.py hardcodes max_episodes_rendered=10 (line ~772), so a
# 50-episode-per-task run only keeps video for the first 10 episodes -- too few
# to review failures that land at higher episode indices. This wrapper forces
# the cap up so EVERY episode gets an MP4. CLI is identical to lerobot-eval,
# plus one env knob:
#   EVAL_RENDER_ALL_N=<int>  -> max_episodes_rendered per task (default 10000)
#
# Works because eval_main calls eval_policy_all through module globals, so
# rebinding the name here intercepts the call; the kwarg is always passed by
# keyword at the call site.
import os

import lerobot.scripts.lerobot_eval as _le

_N = int(os.environ.get('EVAL_RENDER_ALL_N', '10000'))
_orig = _le.eval_policy_all


def _patched(*args, **kwargs):
    kwargs['max_episodes_rendered'] = _N
    return _orig(*args, **kwargs)


_le.eval_policy_all = _patched
print(f'[fullvideo] max_episodes_rendered forced to {_N}')

if __name__ == '__main__':
    _le.main()
