"""EXPERIMENTAL / DEMO ONLY - not for scored submission.

GemmaController wraps the team's scored controller (controllers/ua_team/pure_pursuit_racing.py,
imported unchanged) and adds a Gemma 4 "driving strategist":

  * every ``strategist_period_s`` (default 1.5 s) the 50 ms step() hands a compact text summary
    of the situation (speed, curvature/obstacles ahead, rear slip, rider request) to a background
    thread and returns immediately;
  * the thread asks Gemma for HIGH-LEVEL settings only: mode (sport|comfort), speed_scale
    (0.6-1.0, never faster than the scored settings) and margin_extra (0-0.3 m extra cone margin);
  * the answer is clamped to those bounds and ramped in gradually; steering, braking and the
    racing line remain the numpy controller's job. If Gemma is unreachable or answers nonsense,
    the scored defaults (sport, 1.0, 0.0) stay in force.

Why it is not a submission: the official harness runs controllers in a sandbox that forbids
network access (and the rules allow numpy only), so inside ``python -m autocross run`` the
strategist cannot reach the model and the controller simply behaves like plain v2. To actually
see Gemma in the loop use ``python gemma/run_demo.py`` (in-process, unsandboxed demo runner).
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "controllers" / "ua_team"))
from pure_pursuit_racing import PurePursuitRacing  # noqa: E402  (scored controller, unchanged)
from llm import LLMUnavailable, get_backend, parse_json  # noqa: E402

STRATEGIST_SYSTEM = """You are the driving strategist for a self-driving EcoCAR (Chevrolet Blazer EV) on an \
autocross course. A real-time controller handles steering and braking. You only choose high-level settings.
Reply with ONE JSON object: {"mode": "sport" or "comfort", "speed_scale": number 0.6-1.0, \
"margin_extra": number 0.0-0.3, "reason": "short sentence"}.
Guidelines: use comfort if the rider asks for a gentle ride or feels unwell; lower speed_scale if the rear \
is sliding (rear_slip_deg above 4) or grip seems low; add margin_extra when obstacles or very tight corners \
(radius below 12 m) are ahead; otherwise sport, 1.0, 0.0. Safety first, but do not slow down without a reason."""

# mode-specific attributes of PurePursuitRacing that comfort mode changes
_MODE_ATTRS = ("comfort", "mu_frac", "dec_frac", "acc_frac", "hid_dec", "rear_ref", "open_margin",
               "a_lon_c", "jerk")


class GemmaController(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c = dict(self.config)
        self.period = float(c.pop("strategist_period_s", 1.5))
        self.timeout = float(c.pop("strategist_timeout_s", 20.0))
        self.rider = str(c.pop("rider_request", "") or "")
        self.backend_name = c.pop("backend", None)
        self.enabled = bool(c.pop("strategist", True))
        self.inner = PurePursuitRacing({**c, "mode": "sport"})
        self.inner.on_start(track, vehicle)
        ref = PurePursuitRacing({**c, "mode": "comfort"})
        ref.on_start(track, vehicle)
        self.profiles = {"sport": {a: getattr(self.inner, a, None) for a in _MODE_ATTRS},
                         "comfort": {a: getattr(ref, a, None) for a in _MODE_ATTRS}}
        self.base_margin = self.inner.margin
        # requested (from Gemma) and applied (ramped) settings
        self.req = {"mode": "sport", "speed_scale": 1.0, "margin_extra": 0.0, "reason": "default"}
        self.mode = "sport"
        self.scale = 1.0
        self.margin_extra = 0.0
        self.status = "idle"            # idle | waiting | ok | unavailable | bad_reply
        self.n_calls = self.n_ok = 0
        self.last_latency_s = float("nan")
        self.log = []                    # (sim time, decision) for the demo runner
        self._lock = threading.Lock()
        self._busy = False
        self._next_t = 0.5
        self._backoff_until = 0.0
        self._backend = get_backend(self.backend_name) if self.enabled else None
        initial = str(c.get("mode", "sport"))
        if initial in self.profiles:
            self.req["mode"] = initial

    # ------------------------------------------------------------- background
    def _summary(self, obs: Observation) -> str:
        r, vp = obs.road, self.vehicle
        kc = np.abs(np.asarray(r.preview_curvature)) if r is not None and r.valid else np.zeros(1)
        rmin = 1.0 / max(float(kc.max()), 1e-3)
        obst = [o for o in (r.obstacles if r is not None else []) if o.distance < 40]
        beta_r = math.degrees(abs(math.atan2(obs.vy - vp.cg_to_rear * obs.yaw_rate, max(obs.vx, 1.0))))
        corners = []
        for cn in (r.corners if r is not None else [])[:3]:
            corners.append(f"{'left' if cn.direction > 0 else 'right'} r={cn.min_radius:.0f}m in {max(cn.entry_distance, 0):.0f}m")
        return (f"t={obs.t:.1f}s speed={obs.speed:.1f}m/s lateral_g={abs(obs.ay) / 9.81:.2f} "
                f"rear_slip_deg={beta_r:.1f} tightest_radius_ahead={rmin:.0f}m corners_ahead=[{'; '.join(corners)}] "
                f"obstacles_ahead={len(obst)} nearest_obstacle_m={min((o.distance for o in obst), default=-1):.0f} "
                f"current_mode={self.mode} speed_scale={self.scale:.2f} margin_extra={self.margin_extra:.2f} "
                f"rider_request=\"{self.rider or 'none'}\"")

    def _ask(self, summary: str, t_sim: float):
        t0 = time.perf_counter()
        try:
            reply = self._backend.chat(STRATEGIST_SYSTEM, summary, json_mode=True, timeout=self.timeout,
                                       max_tokens=120)
            d = parse_json(reply)
            mode = str(d.get("mode", "sport")).lower()
            dec = {"mode": mode if mode in self.profiles else "sport",
                   "speed_scale": float(np.clip(float(d.get("speed_scale", 1.0)), 0.6, 1.0)),
                   "margin_extra": float(np.clip(float(d.get("margin_extra", 0.0)), 0.0, 0.3)),
                   "reason": str(d.get("reason", ""))[:160]}
            with self._lock:
                self.req, self.status = dec, "ok"
                self.n_ok += 1
                self.log.append((t_sim, round(time.perf_counter() - t0, 2), dec))
        except LLMUnavailable as e:
            with self._lock:
                self.status = "unavailable"
                self._backoff_until = t_sim + 10.0
                self.log.append((t_sim, round(time.perf_counter() - t0, 2), {"error": str(e)[:160]}))
        except (ValueError, TypeError) as e:
            with self._lock:
                self.status = "bad_reply"
                self.log.append((t_sim, round(time.perf_counter() - t0, 2), {"error": f"bad reply: {e}"[:160]}))
        finally:
            with self._lock:
                self.last_latency_s = time.perf_counter() - t0
                self._busy = False

    # -------------------------------------------------------------- 50 ms loop
    def _apply(self, dt: float):
        with self._lock:
            req = dict(self.req)
        if req["mode"] != self.mode:
            self.mode = req["mode"]
            for a, v in self.profiles[self.mode].items():
                setattr(self.inner, a, v)
            if self.mode == "comfort":
                self.inner.prev_a = 0.0
        # ramp continuous settings (max 0.1 scale/s, 0.1 m/s) so the plan never jumps
        self.scale += float(np.clip(req["speed_scale"] - self.scale, -0.1 * dt, 0.1 * dt))
        self.margin_extra += float(np.clip(req["margin_extra"] - self.margin_extra, -0.1 * dt, 0.1 * dt))
        prof = self.profiles[self.mode]
        self.inner.mu_frac = prof["mu_frac"] * self.scale
        self.inner.dec_frac = prof["dec_frac"] * self.scale
        self.inner.acc_frac = prof["acc_frac"] * self.scale
        self.inner.margin = self.base_margin + self.margin_extra

    def step(self, obs: Observation) -> Command:
        dt = self.vehicle.control_dt
        if self.enabled and obs.t >= self._next_t and obs.t >= self._backoff_until:
            with self._lock:
                busy = self._busy
                if not busy:
                    self._busy = True
                    self.status = "waiting"
                    self.n_calls += 1
            if not busy:
                self._next_t = obs.t + self.period
                try:
                    threading.Thread(target=self._ask, args=(self._summary(obs), obs.t), daemon=True).start()
                except (RuntimeError, PermissionError):
                    self.enabled = False
        self._apply(dt)
        cmd = self.inner.step(obs)
        cmd.debug.update({"g_comfort": float(self.mode == "comfort"), "g_scale": self.scale,
                          "g_margin": self.margin_extra, "g_ok": float(self.status == "ok"),
                          "g_calls": float(self.n_calls)})
        return cmd
