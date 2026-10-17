"""EXPERIMENTAL demo runner: drives GemmaController with the official simulator, rules, scoring and
logging, but IN-PROCESS (no sandbox), so the strategist thread may talk to Gemma over HTTP.
Scores from this runner are for demonstration only - the official harness sandboxes controllers
and forbids network access.

    python gemma/run_demo.py -t tracks/practice_technical.yaml [--rider "I feel carsick"] [--view --realtime]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import autocross.runner as runner  # noqa: E402
from autocross.config import load_sim_config, load_yaml  # noqa: E402
from autocross.sandbox import StepStats, load_controller_class  # noqa: E402

CTRL_PATH = str(Path(__file__).resolve().parent / "gemma_controller.py")


class InProcessController:
    """Drop-in for autocross.sandbox.ControllerProcess (same interface), without the sandbox."""
    instance = None

    def __init__(self, path, budget, class_name=None, ctrl_cfg=None, protected_paths=None):
        cls = load_controller_class(path, class_name)
        self.ctrl = cls(ctrl_cfg or {})
        self.deadline_ms = float(budget.get("step_deadline_ms", 50.0))
        self.dead = False
        self.peak_rss = 0
        self.drive_cpu_s = float("nan")
        self.init_ms = float("nan")
        self.step_ms = []
        InProcessController.instance = self

    def start(self, info, vp):
        t0 = time.perf_counter()
        self.ctrl.on_start(info, vp)
        self.init_ms = (time.perf_counter() - t0) * 1e3
        self._cpu0 = time.thread_time()

    def step(self, obs):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            cmd = self.ctrl.step(obs)
        except Exception as e:  # noqa: BLE001
            return None, StepStats(error=repr(e), overrun=True)
        ms = (time.perf_counter() - t0) * 1e3
        self.step_ms.append(ms)
        return cmd, StepStats(ms, (time.thread_time() - c0) * 1e3, ms, ms > self.deadline_ms)

    def close(self):
        self.drive_cpu_s = time.thread_time() - getattr(self, "_cpu0", time.thread_time())
        try:
            import psutil
            self.peak_rss = psutil.Process().memory_info().rss
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-t", "--track", default=str(ROOT / "tracks" / "practice_technical.yaml"))
    ap.add_argument("--rider", default="", help='rider request passed to the strategist, e.g. "I feel carsick"')
    ap.add_argument("--controller-config", help="YAML with controller settings (e.g. gemma/rider_config.yaml)")
    ap.add_argument("--backend", help="google (default) | ollama | mock")
    ap.add_argument("--period", type=float, default=1.5, help="seconds of sim time between strategist calls")
    ap.add_argument("--realtime", nargs="?", type=float, const=1.0, help="pace to wall clock (recommended "
                    "with a real model, so its 1-2 s answers arrive at the right point of the lap)")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-strategist", action="store_true", help="plain v2 through the same runner")
    a = ap.parse_args()

    cfg = load_sim_config()
    if a.realtime:
        cfg["sim"]["realtime_factor"] = a.realtime
    ccfg = load_yaml(a.controller_config) if a.controller_config else {}
    ccfg.update({"strategist_period_s": a.period, "strategist": not a.no_strategist})
    if a.rider:
        ccfg["rider_request"] = a.rider
    if a.backend:
        ccfg["backend"] = a.backend
    viewer = None
    if a.view:
        from autocross.track import load
        from autocross.viewer import TopDownViewer
        viewer = TopDownViewer(load(a.track))
    runner.ControllerProcess = InProcessController   # demo only: no sandbox
    try:
        s = runner.run(cfg, CTRL_PATH, a.track, backend="bicycle", controller_cfg=ccfg, team="gemma_demo",
                       tag="gemma", viewer=viewer, verbose=True)
    finally:
        if viewer:
            viewer.close()
    ip = InProcessController.instance
    g = ip.ctrl
    import numpy as np
    st = np.asarray(ip.step_ms)
    print(f"\nGemma strategist: backend={getattr(g._backend, 'name', None)} calls={g.n_calls} ok={g.n_ok} "
          f"status={g.status}")
    print(f"50 ms loop (in-process step time): p50 {np.percentile(st, 50):.2f} ms, p99 {np.percentile(st, 99):.2f} ms, "
          f"max {st.max():.2f} ms, steps over 50 ms: {(st > 50).sum()}")
    for t, lat, d in g.log:
        print(f"  t={t:5.1f}s  ({lat:.1f}s)  {json.dumps(d)}")


if __name__ == "__main__":
    main()
