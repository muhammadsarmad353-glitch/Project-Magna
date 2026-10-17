# Gemma 4 add-on (EXPERIMENTAL / DEMO, not the scored submission)

This folder adds Google's **Gemma 4** open models *around* our scored controller
(`controllers/ua_team/pure_pursuit_racing.py`, v2 + comfort mode). It does not change that controller.
The add-on only imports it.

| File | What it does |
|---|---|
| `llm.py` | Pluggable backend, stdlib only. **Google AI Studio / Gemini API (default)** → local **Ollama** fallback → `mock`. |
| `gemma_controller.py` | `GemmaController` wraps `PurePursuitRacing`. An async background-thread "strategist" runs about every 1.5 s. It picks only `mode` (sport/comfort), `speed_scale` (0.6–1.0) and `margin_extra` (0–0.3 m), and changes are ramped in. The 50 ms loop never waits for the model. If the model is unreachable, the controller keeps the scored defaults. |
| `run_demo.py` | Demo runner. It uses the official simulator, scoring and logging, but runs the controller **in-process (unsandboxed)** so the strategist can reach Gemma. |
| `narrator.py` | Reads a run log (`runs/...`). It prints a passenger narration and a short engineering summary of where time, cones or points were lost. |
| `voice_mode.py` | Text "voice mode". It maps rider requests ("I feel carsick", "I'm late") to a comfort/sport config and writes `gemma/rider_config.yaml`. |

## How this relates to the scored controller
* **The scored submission stays `controllers/ua_team/pure_pursuit_racing.py`.** No LLM is in the scored loop. The rules allow only numpy on 1 core with no network, step p99 must stay under 5 ms, and an LLM needs 0.5–8 s per answer.
* The official harness runs controllers in a sandbox that blocks sockets. If you run `gemma_controller.py` with `python -m autocross run`, the strategist cannot reach the model, so it falls back. The result is exactly plain v2 (tested: 34.05 s, 71.3/75).
* `run_demo.py` is the only way to see Gemma actually steering the strategy. Its scores are **demo-only**.
* `narrator.py` and `voice_mode.py` run offline, before or after a run, and never touch the scored loop.

## What Gemma 4 is
* Google DeepMind's open-weight model family (Apache 2.0), released 31 Mar 2026. It is multimodal (text and image; small models also take audio) and supports function calling and JSON output.
* Sizes: **E2B** and **E4B** (on-device, 128K context), **26B A4B** MoE (about 4B active), **31B** dense (256K context), plus a 12B "unified" model (Jun 2026).
* Gemini API model ids: `gemma-4-26b-a4b-it` (default here) and `gemma-4-31b-it`.
* Ollama tags: `gemma4:e2b` (about 7 GB), `gemma4:e4b` (= `gemma4`, about 10 GB), `gemma4:26b`, `gemma4:31b`.

## Windows setup (PowerShell)
```powershell
cd ecocar-hackathon-sim
git checkout gemma4            # or unzip gemma4-addon.zip into the repo root
.\.venv\Scripts\Activate.ps1   # same venv as the simulator; no extra packages needed

# --- Default backend: Google AI Studio / Gemini API ---
# get a key at https://aistudio.google.com/apikey
$env:GEMINI_API_KEY = "your-key-here"
$env:GEMMA_BACKEND  = "google"                 # default anyway; falls back to Ollama if the API fails
# optional: $env:GOOGLE_MODEL = "gemma-4-31b-it"   (default gemma-4-26b-a4b-it)

# --- Optional fallback: local Ollama (offline) ---
# install from https://ollama.com/download/windows, then:
ollama pull gemma4:e4b                         # or gemma4:e2b on laptops with < 16 GB RAM
$env:OLLAMA_MODEL = "gemma4:e4b"
# $env:GEMMA_BACKEND = "ollama"                # force Ollama only

# --- No model at all ---
$env:GEMMA_BACKEND = "mock"                    # deterministic rules, for testing
```
To make the key permanent, run `setx GEMINI_API_KEY "your-key-here"` and then open a new terminal. Never commit the key.

## Run
```powershell
# 1) Narrate a run (newest in runs/, or give a folder)
python gemma\narrator.py
python gemma\narrator.py runs\<run-folder> --facts

# 2) Voice mode -> config for the normal (scored) controller
python gemma\voice_mode.py "I feel carsick"
python -m autocross run -c controllers\ua_team\pure_pursuit_racing.py -t tracks\practice_technical.yaml --backend bicycle --controller-config gemma\rider_config.yaml --view --realtime

# 3) Gemma strategist in the loop (DEMO runner, unsandboxed)
python gemma\run_demo.py -t tracks\practice_technical.yaml --realtime          # real-time so answers land on time
python gemma\run_demo.py --rider "I feel carsick" --realtime --view
python gemma\run_demo.py --no-strategist                                        # plain v2 through same runner

# 4) Same wrapper in the OFFICIAL sandboxed harness (falls back -> identical to v2)
python -m autocross run -c gemma\gemma_controller.py -t tracks\practice_technical.yaml --backend bicycle
```
`run_demo.py` prints the strategist's decisions with their latency, plus step-time p50/p99/max. Use these numbers to check that the 50 ms loop is unaffected.

## Safety design of the strategist
* Gemma cannot steer, brake, or pick the line. It can only pick within `speed_scale ∈ [0.6, 1.0]` and `margin_extra ∈ [0, 0.3] m`, plus a mode switch. Every value is clamped, and changes are ramped at 0.1 per second.
* The model call runs on a daemon thread. `step()` only reads the latest decision under a lock, so the step time is unaffected. If a request is still in flight, the next one is skipped.
* An error or timeout keeps the last good decision (initially the scored defaults) and backs off for 10 s. A malformed reply is ignored.
