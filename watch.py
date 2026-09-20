#!/usr/bin/env python3
"""
apprentice-watch — pings your phone when an apprenticeship application opens.

For each firm in firms.json it fetches the careers / apprenticeship page,
works out whether applications are OPEN right now, and sends a phone push the
moment a firm flips from not-open to open. State is kept in state/seen.json so
you get pinged ONCE per opening, not every run.

Runs free on GitHub Actions — see .github/workflows/check.yml.

Classification:
  - If GEMINI_API_KEY is set, Google Gemini reads the page text and decides
    open / register-interest / closed / unknown (handles nuance keywords miss).
    Uses the free Google AI Studio tier — no billing needed.
  - If not, a keyword fallback is used (free, no API key, slightly dumber).
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ---------------- config ----------------
FIRMS_FILE = Path("firms.json")
STATE_FILE = Path("state/seen.json")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")                 # REQUIRED
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")         # optional (smarter than keywords)
# Model names change often and vary by project, so instead of hardcoding one we
# ask the API which models this key can actually use and pick the best available.
GEMINI_MODEL = None                                       # resolved at runtime (see resolve_model)
GEMINI_MODEL_PREFS = ["gemini-2.5-flash-lite", "gemini-2.5-flash",
                      "gemini-flash-latest", "gemini-2.0-flash"]
REQUEST_SPACING = 5.0        # seconds between firms — polite to sites AND keeps
                             # us under Gemini's free per-minute request limit
UA = "Mozilla/5.0 (apprentice-watch; personal apprenticeship alert)"
TIMEOUT = 30

OPEN, REGISTER, CLOSED, UNKNOWN = "open", "register_interest", "closed", "unknown"


def fetch(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def visible_text(html):
    """Crude strip of tags/scripts/styles -> visible text."""
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):
    """Stable hash of a page's visible text — lets us skip the (paid) AI
    classify when a page hasn't changed since the last run."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_keywords(text):
    t = text.lower()
    if any(k in t for k in [
        "applications are closed", "application closed", "no longer accepting",
        "closed for applications", "you can no longer apply", "applications have closed",
    ]):
        return CLOSED
    if any(k in t for k in [
        "apply now", "start your application", "apply here", "start application",
        "submit your application", "applications are open", "apply for this role",
    ]):
        return OPEN
    if any(k in t for k in [
        "register your interest", "register interest", "notify me", "coming soon",
        "opening soon", "sign up for alerts", "join our talent",
    ]):
        return REGISTER
    return UNKNOWN


def resolve_model():
    """Ask the API which models this key can use for generateContent and pick the
    best available (cheapest/fastest first). Avoids hardcoding a name that 404s."""
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": GEMINI_API_KEY}, timeout=TIMEOUT)
        r.raise_for_status()
        usable = {
            m["name"].split("/")[-1]
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }
        for pref in GEMINI_MODEL_PREFS:
            if pref in usable:
                return pref
        flash = sorted(x for x in usable if "flash" in x)   # any flash model
        if flash:
            return flash[0]
        if usable:
            return sorted(usable)[0]                         # last resort: anything usable
    except Exception as e:
        print(f"  model discovery failed ({e}); defaulting", file=sys.stderr)
    return "gemini-2.5-flash"


def classify_ai(text):
    snippet = text[:6000]   # keep the call cheap
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    prompt = (
        "You are checking a company careers page for a DEGREE APPRENTICESHIP "
        "(school-leaver / pre-university) role. Based ONLY on the text, can an "
        "application be SUBMITTED right now?\n"
        'Return strict JSON, no prose: '
        '{"status": "open"|"register_interest"|"closed"|"unknown"}\n'
        "- open = you can submit an application now\n"
        "- register_interest = only 'register interest' / 'notify me' / 'coming soon'\n"
        "- closed = applications closed or expired\n"
        "- unknown = cannot tell / page looks empty\n\n"
        f"PAGE TEXT:\n{snippet}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 200,
            "responseMimeType": "application/json",   # force clean JSON, no ``` fences
        },
    }
    for attempt in range(2):   # one retry if the free tier rate-limits us
        try:
            r = requests.post(url, params={"key": GEMINI_API_KEY},
                              json=body, timeout=TIMEOUT)
            if r.status_code == 429 and attempt == 0:
                time.sleep(20)          # hit the per-minute limit — wait, then retry once
                continue
            r.raise_for_status()
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = re.sub(r"^```json|```", "", raw.strip()).strip()
            status = json.loads(raw).get("status", UNKNOWN)
            return status if status in (OPEN, REGISTER, CLOSED, UNKNOWN) else UNKNOWN
        except Exception as e:
            print(f"  ai classify failed ({e}); using keywords", file=sys.stderr)
            return classify_keywords(text)
    return classify_keywords(text)


def classify(text):
    if not text or len(text) < 200:
        return UNKNOWN   # near-empty = probably a JS shell we can't read
    return classify_ai(text) if GEMINI_API_KEY else classify_keywords(text)


def notify(title, message, url=None, priority="default", tags="briefcase"):
    if not NTFY_TOPIC:
        print("  NTFY_TOPIC not set; would notify:", title, "-", message, file=sys.stderr)
        return
    # HTTP headers must be latin-1, so an emoji in the title crashes the send.
    # ntfy already renders the Tags value as an emoji in front of the title,
    # so we just strip any non-latin-1 chars (emoji) out of the Title header.
    safe_title = title.encode("latin-1", "ignore").decode("latin-1").strip()
    headers = {"Title": safe_title, "Priority": priority, "Tags": tags}
    if url:
        headers["Click"] = url
    try:
        requests.post(f"{NTFY_SERVER}/{NTFY_TOPIC}",
                      data=message.encode("utf-8"), headers=headers, timeout=TIMEOUT)
    except Exception as e:
        print(f"  notify failed: {e}", file=sys.stderr)


def main():
    if GEMINI_API_KEY:
        global GEMINI_MODEL
        GEMINI_MODEL = resolve_model()
        print(f"Using Gemini model: {GEMINI_MODEL}")

    firms = json.loads(FIRMS_FILE.read_text())
    first_run = not STATE_FILE.exists()
    state = {} if first_run else json.loads(STATE_FILE.read_text())
    new_state = dict(state)
    missing = []

    for firm in firms:
        name = firm["name"]
        url = firm.get("url", "").strip()
        if not url:
            missing.append(name)
            continue

        prev_entry = state.get(name, {})
        prev = prev_entry.get("status", UNKNOWN)
        prev_hash = prev_entry.get("hash")
        try:
            text = visible_text(fetch(url))
        except Exception as e:
            print(f"[{name}] fetch error: {e}", file=sys.stderr)
            if prev != "error":                                 # fail loud, once
                notify(f"⚠️ Can't check {name}",
                       f"Couldn't read {url}. Check it manually / fix the URL.",
                       url=url, tags="warning")
            new_state[name] = {"status": "error"}
            time.sleep(REQUEST_SPACING)
            continue

        h = text_hash(text)
        if h == prev_hash and prev in (OPEN, REGISTER, CLOSED, UNKNOWN):
            status = prev                                       # page unchanged -> reuse, no AI call
            print(f"[{name}] {prev} -> {status} (unchanged, skipped)")
        else:
            status = classify(text)                             # page changed -> classify (may use AI)
            print(f"[{name}] {prev} -> {status}")

        new_state[name] = {"status": status, "hash": h}

        if status == OPEN and prev != OPEN and not first_run:           # the event we want
            notify(f"🎓 {name} apprenticeship is OPEN",
                   f"{name} is accepting applications now. Apply fast — rolling closes.",
                   url=url, priority="high", tags="rotating_light")
        elif status == UNKNOWN and prev != UNKNOWN and not first_run:   # can't read it -> tell me once
            notify(f"❓ Can't judge {name}",
                   f"{name}'s page changed but I can't tell if it's open "
                   f"(often a JS-only portal). Worth a manual look.",
                   url=url, tags="grey_question")

        time.sleep(REQUEST_SPACING)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(new_state, indent=2))

    if first_run:
        unknowns = sum(1 for v in new_state.values() if v.get("status") == UNKNOWN)
        notify("✅ apprentice-watch is live",
               f"Baseline saved for {len(firms) - len(missing)} firms "
               f"({unknowns} unreadable for now). "
               "You'll be pinged when any opens.", tags="white_check_mark")
    if missing:
        print("No URL set for:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
