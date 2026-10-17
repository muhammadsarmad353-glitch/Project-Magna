# UA Team: pure pursuit on a planned racing line

`pure_pursuit_racing.py` (class `PurePursuitRacing`) is our HackBama EcoCAR autocross controller.
It uses only numpy and `autocross.api` and plans everything from `obs.road`, with no map.
`pure_pursuit_racing_v1.py` is the earlier v1, kept as a backup and reference.

## Design

Every 50 ms control step:

1. **Corridor.** The road edges are shrunk by half the car width (1.08 m) plus a safety margin.
   The margin is 0.55 m, plus 4 m x |curvature| for tight corners (the body corners swing out), plus 0.03 m per m/s above 15 m/s (tracking lag at speed).
   Each obstacle blocks the side it sits on over a ±4 m window, and the car passes on the side with the larger gap (0.45 m from the obstacle cones).
   Near the car the corridor is blended to include the car's actual offset, so re-entry is smooth instead of kinked.
2. **Minimum-curvature racing line.** Lateral offsets n(s) from the centerline minimise Σ(κ_centerline + n'')².
   This is a 41-point banded least-squares problem with corridor bounds handled by a few active-set iterations.
   The plan is warm-started and kept consistent with the previous plan, because path deviation is scored against the path published 0.5 s earlier.
   A kink guard smooths the published path if it would fail the harness's plausibility check; a failed path is discarded and scored against the centerline instead.
3. **Friction-circle speed plan.** v ≤ sqrt(a_lat / |κ|) with a_lat = 0.83·μg, followed by a backward braking pass and a forward acceleration pass inside the friction circle.
   It also checks a worst case: a 9 m radius hairpin hidden just past the 40 m preview, which the car only needs to be able to stop for at a firmer braking level.
4. **Lateral control: pure pursuit on the racing line (not the centerline).**
   - Look-ahead Ld = clip(0.20·v + 2, 4, 12) m, measured from a pose predicted 0.05 s ahead (steering lag).
   - δ = atan(L·κ_pp) + 0.003·v²·κ_pp (understeer compensation).
   - Steering is capped so it never asks for more than the available grip (the grip-demand rule).
   - The steering rate is limited to 0.98 × the actuator's 2.0/s.
5. **Longitudinal control.** Feed-forward acceleration from the speed plan plus P feedback, with friction-circle caps on throttle and brake.
   The car slows down whenever pure pursuit asks for more curvature than the grip allows at the current speed.
   Throttle and brake are never applied together.
6. **Grip observer and stability control.** If the rear slip angle goes above 4° the usable grip is reduced (down to 80%).
   Above 4.5° the brake is eased and throttle is cut (ESC-style). This fixed a spin-out at 10% less grip.
7. **Drive modes.** `mode: sport` (default, the scored settings) or `mode: comfort`.
   Comfort is a consumer/passenger mode: about 0.35 g lateral, 0.3 g longitudinal, and a 2.5 m/s³ jerk limit.
   It may brake up to about 0.6 g only when needed to make the next corner safely.
   In comfort mode, pure pursuit is referenced to the rear axle and obstacle passing may use more of the open side.

All parameters can be overridden with `--controller-config file.yaml` (see `on_start`).

## How to run

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
python -m autocross selftest -c controllers\ua_team\pure_pursuit_racing.py
# watch at 1x real time (without --realtime the viewer runs as fast as the PC can simulate, ~10x)
python -m autocross run -c controllers\ua_team\pure_pursuit_racing.py -t tracks\practice_technical.yaml --backend bicycle --view --realtime
# comfort mode
python -m autocross run -c controllers\ua_team\pure_pursuit_racing.py -t tracks\practice_technical.yaml --backend bicycle --view --realtime --controller-config comfort.yaml
```

Linux/macOS:
```bash
source .venv/bin/activate
python -m autocross selftest -c controllers/ua_team/pure_pursuit_racing.py
python -m autocross run -c controllers/ua_team/pure_pursuit_racing.py -t tracks/practice_technical.yaml --backend bicycle --view --realtime
python -m autocross run -c controllers/ua_team/pure_pursuit_racing.py -t tracks/practice_technical.yaml --backend bicycle --view --realtime --controller-config comfort.yaml
# random courses we tested on
python -m autocross run -c controllers/ua_team/pure_pursuit_racing.py -t tracks/generated/gen_42.yaml --backend bicycle
```

`comfort.yaml` (also at the repo root):
```yaml
mode: comfort
comfort_lat_g: 0.35
comfort_lon_g: 0.30
comfort_jerk: 2.5
```

The generated courses (`gen-track --seed N --obstacles 3`) are in `tracks/generated/`.
`selftest` only globs `tracks/*.yaml`, so it still runs just the 5 public courses; pass `-t tracks/generated/...` to `run`.
Seed 13 produces the same layout as `hackathon_test`.

## Results (bicycle backend, official settings)

Technical points out of 75 (completion time 30, path deviation 15, compute 15, dynamics 15, minus penalties).
The 25 innovation points are judged separately.

### Public courses

| Course | Baseline | v1 | v2 (final) | v2 time ratio | v2 deviation mean/max (m) | Cones (base / v1 / v2) |
|---|---|---|---|---|---|---|
| practice_technical | 57.5 | 70.65 | **71.33** | 1.078 | 0.102 / 0.35 | 4 / 0 / 0 |
| hackathon_test | 61.9 | 69.04 | **71.17** | 1.079 | 0.098 / 0.34 | 2 / 0 / 0 |
| practice_obstacles | 59.5 | 70.31 | **71.13** | 1.074 | 0.133 / 0.41 | 3 / 0 / 0 |
| practice_basic | 66.2 | 71.26 | **71.71** | 1.065 | 0.101 / 0.37 | 0 / 0 / 0 |
| practice_loop | 66.5 | 70.07 | **71.16** | 1.084 | 0.092 / 0.46 | 0 / 0 / 0 |
| **Mean** | 62.3 | 70.27 | **71.30** | | | |

- **v2 per category:** time 26.2–26.7 / 30, deviation 14.7–15.0 / 15, compute 15 / 15, dynamics 14.85–15 / 15.
- **Penalties:** 0 off-course, 0 missed gates and 0 DNFs for v1 and v2. All baseline cone hits were obstacle cones.
- **Weighted score** (technical course 60%, other public courses 40%): baseline 59.9, v1 70.46, v2 **71.31**.

### Generated courses (10 seeds: 3, 5, 7, 11, 13, 19, 23, 29, 31, 42; 3 obstacles each)

| | Baseline (6 seeds) | v1 | v2 |
|---|---|---|---|
| Technical points | 57.5–62.2 | 69.7–71.2 | 70.0–71.8 (mean ≈ 71.0) |
| Cones / off / gates / DNF | 2–4 cones per course | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Time ratio | 1.22–1.25 | 1.06–1.10 | 1.054–1.095 |

### Robustness (v2)

32 runs: 4 perturbations x 8 courses (5 public plus seeds 19, 3 and 31).
- Perturbations: 1-step actuator delay, `friction_scale=0.9`, 1 m lateral start offset, 5° start heading offset.
- Every run finished with 0 cones, 0 off-course and 0 missed gates (68.0–71.5 points).
- v1 DNF'd on `practice_loop` at `friction_scale=0.9` (rear lock-up under braking); v2's stability control fixes this.

### Comfort mode (practice_technical)

- **Lap:** 46.9 s (1.49x reference), 59.2 / 75, 0 cones.
- **Speeds and accelerations:** top speed 19 m/s (42 mph), lateral 0.38 g p95 (0.51 g peak), acceleration 0.31 g peak, braking up to 0.67 g (safety override).
- **For comparison, sport mode:** top speed 22.2 m/s (50 mph), corners at a median 0.59 g (0.82 g peak), braking 0.84 g.
  That is close to what a real car on road tyres can do (about 0.8–0.9 g) and well above passenger comfort (about 0.3–0.4 g).

## Known limitations

- **Lap time** is 1.05–1.10x the reference (full marks at 0.95x). Pushing grip or braking harder brought back cones or slides somewhere.
  The next gain needs better corner-entry tracking, not just more grip.
- **Comfort mode** hit 2 obstacle cones across the 15 courses (`practice_obstacles` and generated seed 11), while squeezing through narrow obstacle gaps at low speed.
  Braking can also exceed 0.3 g when the safety override is needed. It is a demo mode and is not tuned for scoring.
- **Step time** was measured at about 1.5 ms p99 on the development box, and about 6.5 ms p99 on a team laptop (Windows).
  Both are well inside the 50 ms deadline, but the compute score depends on the machine (full marks at ≤ 5 ms p99).
- **CARLA:** we developed and scored on the bicycle backend only; the official evaluation counts CARLA for half. Results were not checked on CARLA.
- Curvature feed-forward made tracking worse in our tests and is off (`k_ff: 0`).
