#!/usr/bin/env python3
"""
apprentice-watch — pings your phone when an apprenticeship application opens.

For each firm in firms.json it fetches the careers / apprenticeship page,
works out whether applications are OPEN right now, and sends a phone push the
moment a firm flips from not-open to open. State is kept in state/seen.json so
you get pinged ONCE per opening, not every run.

Runs free on GitHub Actions — see .github/workflows/check.yml.

Classification:
  - If ANTHROPIC_API_KEY is set, Claude reads the page text and decides
    open / register-interest / closed / unknown (handles nuance keywords miss).
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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")   # optional (better)
REQUEST_SPACING = 2.0        # seconds between firms — be polite, avoid blocks
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


def classify_ai(text):
    snippet = text[:6000]   # keep the call cheap
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",   # cheap + fast, fine for this
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "You are checking a company careers page for a DEGREE APPRENTICESHIP "
                    "(school-leaver / pre-university) role. Based ONLY on the text, can an "
                    "application be SUBMITTED right now?\n"
                    'Return strict JSON, no prose: '
                    '{"status": "open"|"register_interest"|"closed"|"unknown", '
                    '"deadline": "<text or null>"}\n'
                    "- open = you can submit an application now\n"
                    "- register_interest = only 'register interest' / 'notify me' / 'coming soon'\n"
                    "- closed = applications closed or expired\n"
                    "- unknown = cannot tell / page looks empty\n\n"
                    f"PAGE TEXT:\n{snippet}"
                ),
            }],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        raw = re.sub(r"^```json|```", "", raw.strip()).strip()
        status = json.loads(raw).get("status", UNKNOWN)
        return status if status in (OPEN, REGISTER, CLOSED, UNKNOWN) else UNKNOWN
    except Exception as e:
        print(f"  ai classify failed ({e}); using keywords", file=sys.stderr)
        return classify_keywords(text)


def classify(text):
    if not text or len(text) < 200:
        return UNKNOWN   # near-empty = probably a JS shell we can't read
    return classify_ai(text) if ANTHROPIC_API_KEY else classify_keywords(text)


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
