"""Pluggable Gemma backends (stdlib only: urllib + json).

    GEMMA_BACKEND = google (default) | ollama | mock
    GOOGLE_MODEL  = Gemini API model id (default gemma-4-26b-a4b-it; also gemma-4-31b-it)
    OLLAMA_MODEL  = Ollama tag (default gemma4:e4b; gemma4:e2b for low-RAM laptops)
    GEMMA_MODEL   = legacy: applies to whichever backend has no specific variable set
    OLLAMA_HOST   = http://localhost:11434 (default)
    GEMINI_API_KEY or GOOGLE_API_KEY = key for the Google AI Studio / Gemini API backend

Every backend has ``chat(system, user, json_mode=False, timeout=...) -> str`` and raises
``LLMUnavailable`` when the model cannot be reached. Callers fall back to safe defaults.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class LLMUnavailable(RuntimeError):
    pass


def _post(url: str, payload: dict, timeout: float, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise LLMUnavailable(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, PermissionError) as e:
        raise LLMUnavailable(str(e)) from e


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL") or os.environ.get("GEMMA_MODEL") or "gemma4:e4b"
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host

    def chat(self, system: str, user: str, json_mode: bool = False, timeout: float = 120.0,
             max_tokens: int = 600) -> str:
        payload = {"model": self.model, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "options": {"temperature": 0.3, "num_predict": max_tokens}}
        if json_mode:
            payload["format"] = "json"
        try:
            out = _post(f"{self.host}/api/chat", payload, timeout)
        except LLMUnavailable as e:
            if "think" in str(e):          # older Ollama / model without a thinking switch
                payload.pop("think")
                out = _post(f"{self.host}/api/chat", payload, timeout)
            else:
                raise
        return out.get("message", {}).get("content", "")


class GoogleBackend:
    """Gemma 4 hosted on the Gemini API (Google AI Studio key)."""
    name = "google"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("GOOGLE_MODEL") or os.environ.get("GEMMA_MODEL") or "gemma-4-26b-a4b-it"
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def chat(self, system: str, user: str, json_mode: bool = False, timeout: float = 60.0,
             max_tokens: int = 600) -> str:
        if not self.key:
            raise LLMUnavailable("set GEMINI_API_KEY or GOOGLE_API_KEY")
        # Gemma on the Gemini API: the system prompt is sent as part of the user turn
        text = f"{system}\n\n{user}" + ("\n\nAnswer with a single JSON object only." if json_mode else "")
        payload = {"contents": [{"role": "user", "parts": [{"text": text}]}],
                   "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        out = _post(url, payload, timeout, headers={"x-goog-api-key": self.key})
        try:
            parts = out["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as e:
            raise LLMUnavailable(f"unexpected response: {str(out)[:200]}") from e
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))


class MockBackend:
    """Deterministic stand-in used for tests when no model is available. It is NOT Gemma:
    it answers with simple rules so the plumbing can be tested end to end."""
    name = "mock"

    def chat(self, system: str, user: str, json_mode: bool = False, timeout: float = 1.0,
             max_tokens: int = 600) -> str:
        u = user.lower()
        if "strategist" in system.lower():
            slip = float(_num(r"rear_slip_deg=([\d.]+)", u, 0.0))
            obst = int(_num(r"obstacles_ahead=(\d+)", u, 0))
            sick = any(w in u for w in ("sick", "nause", "slow down", "scared", "comfort"))
            out = {"mode": "comfort" if sick else "sport",
                   "speed_scale": 0.9 if slip > 4 else 1.0,
                   "margin_extra": 0.1 if obst else 0.0,
                   "reason": "mock rules: rider request / slip / obstacles"}
            return json.dumps(out)
        if "rider" in system.lower():
            sick = any(w in u for w in ("sick", "nause", "slow", "scared", "gentle", "smooth", "baby", "coffee"))
            late = any(w in u for w in ("late", "hurry", "fast", "quick", "rush"))
            mode = "comfort" if sick or not late else "sport"
            return json.dumps({"mode": mode, "speed_scale": 0.8 if sick else 1.0,
                               "margin_extra": 0.1 if sick else 0.0,
                               "reply": ("Okay, switching to a smoother, gentler ride." if mode == "comfort"
                                         else "Got it, sport mode. I'll still stay inside the safety limits.")})
        return ("[mock narration - no Gemma model reachable]\n"
                "PASSENGER: We pulled away smoothly, picked up speed on the straights, slowed early for "
                "the tight hairpins and threaded past the orange obstacle cones without touching them.\n"
                "ENGINEERING: see the computed facts above; the mock backend does not analyse them.")


def _num(pat, s, default):
    m = re.search(pat, s)
    return m.group(1) if m else default


class FallbackBackend:
    """Try backends in order (default: Google AI Studio, then local Ollama)."""

    def __init__(self, backends):
        self.backends = backends
        self.name = "+".join(b.name for b in backends)
        self.last_used = None

    def chat(self, *a, **k):
        errs = []
        for b in self.backends:
            try:
                out = b.chat(*a, **k)
                self.last_used = b.name
                return out
            except LLMUnavailable as e:
                errs.append(f"{b.name}: {e}")
        raise LLMUnavailable("; ".join(errs))


def get_backend(name: str | None = None):
    """google (default, falls back to Ollama), ollama, or mock."""
    name = (name or os.environ.get("GEMMA_BACKEND", "google")).lower()
    if name == "mock":
        return MockBackend()
    if name == "ollama":
        return OllamaBackend()
    if name == "google-only":
        return GoogleBackend()
    return FallbackBackend([GoogleBackend(), OllamaBackend()])


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates ```json fences)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    return json.loads(m.group(0))
