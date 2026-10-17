"""Gemma 4 run narrator (offline, post-run - not part of the scored loop).

Reads a run folder written by the simulator (runs/<stamp>_<team>_<track>/) and prints
  1. a passenger-style narration of the lap, and
  2. a short engineering summary of where time / cones / points were lost.
Facts are computed with numpy from summary.json, events.json and telemetry.csv; Gemma only
turns them into prose. If no model is reachable a template text is produced instead.

    python gemma/narrator.py                 # newest run in runs/
    python gemma/narrator.py runs/<folder>   # specific run   [--backend google|ollama|mock] [--facts]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import LLMUnavailable, get_backend  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
G = 9.81


def find_run(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if (p / "summary.json").exists():
            return p
        subs = sorted(p.rglob("summary.json"), key=lambda q: q.stat().st_mtime)
        if subs:
            return subs[-1].parent
        raise SystemExit(f"no summary.json under {p}")
    subs = sorted((ROOT / "runs").rglob("summary.json"), key=lambda q: q.stat().st_mtime)
    if not subs:
        raise SystemExit("no runs found - run the simulator first")
    return subs[-1].parent


def load_telemetry(path: Path) -> dict:
    with open(path) as f:
        cols = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    data = np.atleast_2d(data)
    return {c: data[:, i] for i, c in enumerate(cols)}


def compute_facts(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    ev = json.loads((run / "events.json").read_text()) if (run / "events.json").exists() else []
    m, sc = s.get("metrics", {}), s.get("score", {})
    comp, pen = m.get("completion", {}), m.get("penalties", {})
    f = {"track": s.get("track", {}).get("name", run.name) if isinstance(s.get("track"), dict) else s.get("track"),
         "finished": comp.get("finished"), "dnf_reason": comp.get("dnf_reason"),
         "time_s": comp.get("time_s"), "reference_time_s": comp.get("reference_time_s"),
         "cone_hits": pen.get("cone_hit", 0), "off_course": pen.get("off_course", 0),
         "missed_gates": pen.get("missed_gate", 0), "obstacle_hits": pen.get("obstacle_hits", 0),
         "category_scores": sc.get("category_scores"), "technical_points": sc.get("technical_points"),
         "deductions": sc.get("deductions"),
         "step_ms_p99": m.get("compute", {}).get("step_ms_p99")}
    if f["time_s"] and f["reference_time_s"]:
        f["time_vs_reference_s"] = round(f["time_s"] - f["reference_time_s"], 2)
    f["events"] = [{"t": round(e["t"], 1), "kind": e["kind"], "progress_m": round(e.get("progress") or 0, 1)}
                   for e in ev if e.get("kind") not in ("start",)]
    tp = run / "telemetry.csv"
    if tp.exists():
        T = load_telemetry(tp)
        v, ay, ax = T["speed"], np.abs(T["ay"]), T["ax"]
        f["top_speed_mps"] = round(float(v.max()), 1)
        f["top_speed_mph"] = round(float(v.max()) * 2.237, 1)
        f["avg_speed_mph"] = round(float(v.mean()) * 2.237, 1)
        f["peak_lateral_g"] = round(float(ay.max()) / G, 2)
        f["peak_braking_g"] = round(float(-ax.min()) / G, 2)
        f["peak_accel_g"] = round(float(ax.max()) / G, 2)
        # slowest corners: local speed minima with real lateral load
        t, prog = T["t"], T["progress"]
        mins = [i for i in range(5, len(v) - 5) if v[i] == v[i - 5:i + 6].min() and ay[i - 5:i + 6].max() > 0.3 * G]
        mins = sorted(mins, key=lambda i: v[i])[:3]
        f["slowest_corners"] = [{"t": round(float(t[i]), 1), "progress_m": round(float(prog[i]), 1),
                                 "speed_mph": round(float(v[i]) * 2.237, 1)} for i in sorted(mins)]
        if "path_err" in T:
            f["max_path_error_m"] = round(float(np.nanmax(np.abs(T["path_err"]))), 2)
        if "boundary_margin" in T:
            f["min_boundary_margin_m"] = round(float(np.nanmin(T["boundary_margin"])), 2)
    return f


SYS_NARRATE = ("You are the cheerful in-car voice of a self-driving EcoCAR (Chevrolet Blazer EV) at an autocross. "
               "Using ONLY the facts given, write a 4-6 sentence narration for the passenger about the lap that "
               "just happened: start, fastest moment, the tightest corners, any cone hits, and the finish. "
               "Use mph. No markdown, no invented facts.")
SYS_ENGINEER = ("You are a race engineer. Using ONLY the facts given, write a short engineering summary "
                "(3-5 bullet points, each starting with '- ') of where time, cones or points were lost and one "
                "concrete thing to try next. Be precise with numbers. No invented facts.")


def template_narration(f: dict) -> str:
    cones = f.get("cone_hits", 0)
    lines = [f"And we're off on {f.get('track')}!",
             f"We topped out at {f.get('top_speed_mph')} mph and pulled up to {f.get('peak_lateral_g')} g in the corners."]
    if f.get("slowest_corners"):
        c = min(f["slowest_corners"], key=lambda c: c["speed_mph"])
        lines.append(f"The tightest turn came {c['t']} s in, where we slowed to {c['speed_mph']} mph.")
    lines.append("Not a single cone touched." if not cones else f"We clipped {cones} cone(s) along the way.")
    lines.append(f"Finished in {f.get('time_s'):.2f} s - thanks for riding!" if f.get("finished")
                 else f"We didn't finish this one ({f.get('dnf_reason')}).")
    return " ".join(lines)


def template_engineering(f: dict) -> str:
    out = []
    if f.get("time_vs_reference_s") is not None:
        out.append(f"- Lap {f['time_s']:.2f} s vs reference {f['reference_time_s']:.2f} s "
                   f"({f['time_vs_reference_s']:+.2f} s); category scores {f.get('category_scores')}.")
    out.append(f"- Penalties: cones {f.get('cone_hits')}, off-course {f.get('off_course')}, missed gates "
               f"{f.get('missed_gates')}, deductions {f.get('deductions')}.")
    if f.get("slowest_corners"):
        out.append("- Slowest corners: " + ", ".join(f"{c['speed_mph']} mph @ {c['progress_m']} m" for c in f["slowest_corners"]))
    out.append(f"- Peak lateral {f.get('peak_lateral_g')} g, braking {f.get('peak_braking_g')} g, "
               f"min boundary margin {f.get('min_boundary_margin_m')} m, step p99 {f.get('step_ms_p99')} ms.")
    return "\n".join(out)


def narrate(run: Path, backend_name: str | None = None) -> tuple[dict, str, str, str]:
    f = compute_facts(run)
    facts = json.dumps(f, default=str)
    try:
        be = get_backend(backend_name)
        if getattr(be, "name", "") == "mock":
            return f, template_narration(f), template_engineering(f), "mock (template text)"
        nar = be.chat(SYS_NARRATE, "Facts: " + facts, max_tokens=300, timeout=120).strip()
        eng = be.chat(SYS_ENGINEER, "Facts: " + facts, max_tokens=300, timeout=120).strip()
        used = getattr(be, "last_used", None) or getattr(be, "name", "?")
    except LLMUnavailable as e:
        nar, eng, used = template_narration(f), template_engineering(f), f"template (model unavailable: {e})"
    return f, nar, eng, used


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?")
    ap.add_argument("--backend")
    ap.add_argument("--facts", action="store_true", help="also print the computed facts")
    a = ap.parse_args()
    run = find_run(a.run)
    f, nar, eng, used = narrate(run, a.backend)
    print(f"Run: {run}\nModel: {used}\n")
    if a.facts:
        print("FACTS\n" + json.dumps(f, indent=2, default=str) + "\n")
    print("PASSENGER NARRATION\n" + nar + "\n\nENGINEERING SUMMARY\n" + eng)


if __name__ == "__main__":
    main()
