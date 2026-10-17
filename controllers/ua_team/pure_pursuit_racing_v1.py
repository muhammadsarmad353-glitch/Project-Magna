"""UA team - Option A: pure pursuit on a planned racing line (not the centerline).

Pipeline (every 50 ms, numpy only, road model only - no map):
  1. Corridor   : road edges shrunk by half the car width + a safety margin (more in tight
                  corners, where the body corners swing out); each obstacle blocks the side it
                  sits on, the car passes on the open side.
  2. Racing line: lateral offsets n(s) from the centerline minimising curvature
                  (kappa_c + n'')^2, bounded to the corridor by a few active-set iterations of a
                  small least-squares solve, warm-started from the previous plan (consistency).
  3. Speed plan : v <= sqrt(a_lat/|kappa|), backward braking / forward accel passes inside the
                  friction circle; a ~9 m radius hairpin is assumed just past the 40 m preview.
  4. Lateral    : pure pursuit on the racing line from a pose predicted ahead by the steering
                  lag, look-ahead Ld = clip(k*v + L0, Ld_min, Ld_max), plus an understeer
                  term k_us*v^2*kappa, capped to the grip available, steer-rate limited.
  5. Longitud.  : feed-forward accel from the speed plan + P feedback, friction-circle caps on
                  throttle and brake, slows down if pure pursuit demands more curvature than
                  the grip allows; never throttle and brake together.
  6. Grip obs.  : usable lateral grip is reduced if the rear axle starts to slide.
"""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81


class PurePursuitRacing(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c, vp = self.config, vehicle
        g = lambda k, d: float(c.get(k, d))
        # corridor
        self.half_w = 0.5 * vp.width
        self.margin = g("margin", 0.55)              # m from boundary cones (car edge)
        self.margin_curv = g("margin_curv", 4.0)     # extra margin per 1/m of curvature
        self.obs_margin = g("obs_margin", 0.45)      # m from obstacle cones (car edge)
        self.obs_window = g("obs_window", 4.0)       # +- m along s where an obstacle blocks
        # racing line
        self.w_curv = g("w_curv", 1.0)
        self.w_cons = g("w_cons", 2e-2)
        self.w_reg = g("w_reg", 1e-5)
        self.w_slope = g("w_slope", 2e-4)
        # speed
        self.mu_frac = g("mu_frac", 0.80)
        self.dec_frac = g("decel_frac", 0.75)
        self.acc_frac = g("accel_frac", 1.0)
        self.v_max = g("v_max", 30.0)
        self.r_hidden = g("hidden_radius", 9.0)
        self.kp_v = g("kp_v", 1.5)
        # lateral
        self.k_ld = g("lookahead_gain", 0.20)
        self.ld0 = g("lookahead_l0", 2.0)
        self.ld_min = g("lookahead_min", 4.0)
        self.ld_max = g("lookahead_max", 12.0)
        self.t_pred = g("t_pred", 0.08)              # steering lag compensation (s)
        self.k_ff = g("k_ff", 0.0)
        self.grip_cap = g("grip_cap", 1.0)
        self.blend_t = g("blend_t", 0.6)
        self.blend_min = g("blend_min", 6.0)
        self.brake_lat = g("brake_lat", 0.9)
        self.brake_cap = g("brake_cap", 0.85)
        self.k_us = g("k_us", 0.002)                 # rad per m/s^2 of lateral accel                   # optional path-curvature feed-forward blend
        self.d_steer = 0.98 * vp.max_steer_rate * vp.control_dt
        self.prev_steer = 0.0
        self.mu_scale = 1.0
        self.prev_n = None                           # (s_abs, n) of the previous plan
        self._mat_cache = {}

    # ------------------------------------------------------------------ helpers
    def _base_matrix(self, K, ds):
        key = (K, round(ds, 3))
        m = self._mat_cache.get(key)
        if m is None:
            D2 = np.zeros((K - 2, K))
            i = np.arange(K - 2)
            D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
            D2 /= ds * ds
            D1 = np.zeros((K - 1, K))
            j = np.arange(K - 1)
            D1[j, j], D1[j, j + 1] = -1.0, 1.0
            D1 /= ds
            H = self.w_curv * D2.T @ D2 + self.w_slope * D1.T @ D1 + self.w_reg * np.eye(K)
            m = (D2, H)
            self._mat_cache[key] = m
        return m

    def _plan_offsets(self, s, kc, lo, hi, n0, n_prev, w_prev):
        K = len(s)
        ds = float(np.mean(np.diff(s)))
        D2, H0 = self._base_matrix(K, ds)
        b0 = -self.w_curv * D2.T @ kc[1:-1]
        diag = w_prev.copy()
        rhs = b0 + w_prev * n_prev
        # pin the start to the car's current lateral position
        W0 = 1e4
        diag[0] += W0
        rhs[0] += W0 * n0
        Wb = 1e5
        act_lo = np.zeros(K, bool)
        act_hi = np.zeros(K, bool)
        n = np.clip(n_prev, lo, hi)
        for _ in range(6):
            d = diag + Wb * (act_lo | act_hi)
            r = rhs + Wb * (act_lo * lo + act_hi * hi)
            A = H0 + np.diag(d)
            n = np.linalg.solve(A, r)
            new_lo = act_lo | (n < lo - 1e-3)
            new_hi = act_hi | (n > hi + 1e-3)
            new_lo[0] = new_hi[0] = False
            if (new_lo == act_lo).all() and (new_hi == act_hi).all():
                break
            act_lo, act_hi = new_lo, new_hi
        n[1:] = np.clip(n[1:], lo[1:], hi[1:])
        return n

    # --------------------------------------------------------------------- step
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 5:
            return Command(brake=1.0)
        v = max(obs.speed, 0.0)
        s = np.asarray(road.preview_s, float)
        cl = np.asarray(road.preview_centerline, float)
        kc = np.asarray(road.preview_curvature, float)
        wl = np.asarray(road.preview_width_left, float)
        wr = np.asarray(road.preview_width_right, float)
        K = len(s)

        # centerline frame
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        nrm = np.column_stack((-t[:, 1], t[:, 0]))

        # 1. corridor
        m = self.margin + self.margin_curv * np.abs(kc)
        hi = wl - self.half_w - m
        lo = -(wr - self.half_w - m)
        for ob in road.obstacles:
            win = np.abs(s - ob.distance) < self.obs_window
            if not win.any():
                continue
            iw = np.flatnonzero(win)
            gap_left = float(np.mean(wl[iw])) - ob.lateral_max
            gap_right = ob.lateral_min + float(np.mean(wr[iw]))
            if gap_left > gap_right:      # pass left of it
                lo[iw] = np.maximum(lo[iw], ob.lateral_max + self.half_w + self.obs_margin)
            else:                         # pass right of it
                hi[iw] = np.minimum(hi[iw], ob.lateral_min - self.half_w - self.obs_margin)
        bad = lo > hi
        if bad.any():
            mid = 0.5 * (lo[bad] + hi[bad])
            lo[bad], hi[bad] = mid, mid

        # 2. racing line (warm-started). Near the car the corridor is widened to include the
        # car's actual offset and blended back over a speed-dependent distance, so a car that is
        # slightly outside the corridor gets a smooth re-entry instead of a kink.
        n0 = float(-np.dot(cl[0], nrm[0]))
        r = np.clip(s / max(self.blend_t * v, self.blend_min), 0.0, 1.0)
        lo = np.minimum(lo, n0 + (lo - n0) * r)
        hi = np.maximum(hi, n0 + (hi - n0) * r)
        s_abs = obs.progress + s
        if self.prev_n is not None:
            ps, pn = self.prev_n
            n_prev = np.interp(s_abs, ps, pn)
            known = s_abs <= ps[-1]
            w_prev = np.where(known, self.w_cons * np.exp(-s / 20.0), 0.0)
        else:
            n_prev = np.clip(np.full(K, n0), lo, hi)
            w_prev = np.zeros(K)
        n = self._plan_offsets(s, kc, lo, hi, n0, n_prev, w_prev)
        self.prev_n = (s_abs, n)
        path = cl + n[:, None] * nrm

        # path curvature (geometric, 3-point circle over a 2-sample stencil, then smoothed)
        kp = np.zeros(K)
        a_, b_, c_ = path[:-4], path[2:-2], path[4:]
        ab, bc, ca = b_ - a_, c_ - b_, a_ - c_
        cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
        den = np.hypot(*ab.T) * np.hypot(*bc.T) * np.hypot(*ca.T)
        kp[2:-2] = 2.0 * cross / np.maximum(den, 1e-6)
        kp[:2], kp[-2:] = kp[2], kp[-3]
        kp = np.convolve(np.pad(kp, 2, mode="edge"), np.ones(5) / 5.0, mode="valid")
        seg = np.hypot(*np.diff(path, axis=0).T)
        sarc = np.concatenate(([0.0], np.cumsum(seg)))

        # grip observer (rear slip)
        if v > 3.0:
            beta_r = math.degrees(abs(math.atan2(obs.vy - vp.cg_to_rear * obs.yaw_rate, max(obs.vx, 1.0))))
            if beta_r > 4.0:
                self.mu_scale = max(0.8, self.mu_scale - 0.02)
            else:
                self.mu_scale = min(1.0, self.mu_scale + 0.002)
        a_lat_max = self.mu_frac * self.mu_scale * vp.mu * G
        a_dec = self.dec_frac * vp.max_decel
        a_acc = self.acc_frac * vp.max_accel

        # 3. speed plan
        vlim = np.minimum(np.sqrt(a_lat_max / np.maximum(np.abs(kp), 1e-4)), self.v_max)
        vlim[-1] = min(vlim[-1], math.sqrt(a_lat_max * self.r_hidden))
        vb = vlim.copy()
        for i in range(K - 2, -1, -1):
            alat = vb[i + 1] ** 2 * abs(kp[i + 1])
            ax = a_dec * math.sqrt(max(1.0 - (alat / a_lat_max) ** 2, 0.05))
            vb[i] = min(vb[i], math.sqrt(vb[i + 1] ** 2 + 2.0 * ax * seg[i]))
        vf = vb.copy()
        vf[0] = min(v, vb[0]) if v > 0.5 else vb[0]
        for i in range(K - 1):
            alat = vf[i] ** 2 * abs(kp[i])
            ax = a_acc * math.sqrt(max(1.0 - (alat / a_lat_max) ** 2, 0.05))
            vf[i + 1] = min(vb[i + 1], math.sqrt(vf[i] ** 2 + 2.0 * ax * seg[i]))
        s_look_v = max(v * 0.25, 1.0)
        v_tgt = float(np.interp(s_look_v, sarc, vb))
        v_tgt = min(v_tgt, float(np.interp(s_look_v, sarc, vf)) if v > 0.5 else v_tgt)
        # feed-forward accel from the plan
        s2 = s_look_v + 2.0
        v2 = float(np.interp(s2, sarc, vb))
        a_ff = (v2 ** 2 - v_tgt ** 2) / (2.0 * 2.0)
        a_cmd = a_ff + self.kp_v * (v_tgt - v)

        # 4. lateral: pure pursuit from predicted pose
        tp = self.t_pred
        psi = obs.yaw_rate * tp
        px = obs.vx * tp
        py = obs.vy * tp + 0.5 * obs.vx * psi
        cps, sps = math.cos(psi), math.sin(psi)
        q = path - np.array([px, py])
        q = np.column_stack((cps * q[:, 0] + sps * q[:, 1], -sps * q[:, 0] + cps * q[:, 1]))
        ld = float(np.clip(self.k_ld * v + self.ld0, self.ld_min, self.ld_max))
        d = np.hypot(q[:, 0], q[:, 1])
        i0 = int(np.argmin(d))
        # first point beyond i0 whose distance from the (predicted) car is >= ld
        cand = np.flatnonzero((d >= ld) & (np.arange(K) >= i0) & (q[:, 0] > 0))
        if len(cand):
            j = int(cand[0])
            if j > 0 and d[j - 1] < ld:
                f = (ld - d[j - 1]) / max(d[j] - d[j - 1], 1e-6)
                tgt = q[j - 1] + f * (q[j] - q[j - 1])
            else:
                tgt = q[j]
        else:
            tgt = q[-1]
        L = max(math.hypot(tgt[0], tgt[1]), 1.0)
        alpha = math.atan2(tgt[1], max(tgt[0], 0.1))
        kappa = 2.0 * math.sin(alpha) / L
        if self.k_ff:
            k_path = float(np.interp(v * tp + 0.5 * ld, sarc, kp))
            kappa = (1.0 - self.k_ff) * kappa + self.k_ff * k_path
        # kinematic pure-pursuit steer + understeer compensation (tyres need slip angle)
        delta = math.atan(vp.wheelbase * kappa) + self.k_us * v * v * kappa
        if v > 3.0:   # never command more steer than the tyres can use (grip-demand rule)
            d_cap = math.atan(self.grip_cap * self.mu_scale * vp.mu * G * vp.wheelbase / (v * v))
            delta = float(np.clip(delta, -d_cap, d_cap))
        steer_des = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        steer = float(np.clip(steer_des, self.prev_steer - self.d_steer, self.prev_steer + self.d_steer))
        self.prev_steer = steer

        # 5. longitudinal. The tracking demand also limits speed: if pure pursuit asks for more
        # curvature than the grip allows at this speed, slow down instead of running wide.
        v_pp = math.sqrt(a_lat_max / max(abs(kappa), 1e-4))
        if v_pp < v_tgt:
            v_tgt = v_pp
            a_cmd = min(a_cmd, self.kp_v * (v_tgt - v))
        if a_cmd >= 0:
            lat_use = min(abs(obs.ay) / (vp.mu * G), 1.0)
            cap = max(math.sqrt(max(1.0 - lat_use ** 2, 0.0)), 0.15)
            throttle, brake = min(a_cmd / vp.max_accel + 0.03 * (v > 1.0), cap), 0.0
        else:
            # combined slip: braking hard while cornering hard unloads the rear and lets it slide
            lat_use = min(abs(obs.ay) / (self.brake_lat * vp.mu * G), 1.0)
            cap = max(self.brake_cap * math.sqrt(max(1.0 - lat_use ** 2, 0.0)), 0.08)
            throttle, brake = 0.0, min(-a_cmd / vp.max_decel, cap)
        if not math.isnan(road.finish_distance) and road.finish_distance < -1.0:
            throttle, brake = 0.0, 0.5

        return Command(steer=steer, throttle=float(np.clip(throttle, 0, 1)), brake=float(np.clip(brake, 0, 1)),
                       path=path, path_frame="vehicle", target_speed=v_tgt,
                       debug={"ld": ld, "n0": n0, "vt": v_tgt, "mu_s": self.mu_scale, "kap": kappa, "vb0": float(vb[0]), "lo10": float(lo[10]), "hi10": float(hi[10]), "n10": float(n[10]), "kc10": float(kc[10]), "nob": len(road.obstacles), "dmax": float(np.max(np.abs(np.diff(n)))), "kpm": float(np.max(np.abs(kp[:20])))})
