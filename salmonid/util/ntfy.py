"""In plain terms: this file just sends a phone/desktop alert when a long computer
job finishes or fails, so the operator does not have to sit and watch an overnight run.
It is a convenience helper only -- it never touches the fish data and produces no number,
table, or figure that appears in the paper.

Tiny ntfy.sh notification sender used by the pipeline runners.

What it does
------------
Fires a single one-shot push notification to a fixed ntfy.sh topic. The long-running
GPU jobs in this repo (detection, training, corpus count runs, figure builds) call this
at the end of a stage so the operator's phone/desktop pings when an unattended overnight
run finishes or fails.

Where it sits in the pipeline
-----------------------------
A cross-cutting *utility*, not a pipeline stage — the runners that drive every stage
(data -> detect -> track -> eval/count -> results -> figures) shell out to it purely to
report progress. It touches none of the analysis data and produces no manuscript number
or figure; it is safe to omit entirely (a failed notification never affects results).

Inputs / outputs
----------------
Inputs: a message body string, plus optional ``--title``/``--priority``/``--tags``.
Output: an HTTP POST to ``{NTFY_URL}/{TOPIC}``. If every attempt fails (e.g. the machine
is offline) the message is appended to a local fallback log instead, so a run summary is
never lost during an internet outage.

Environment
-----------
Runs in the base env. Pure Python standard library only (``urllib``) — no torch, no
boxmot, and no dependency on the DEIMv2 ``.venv_deim`` — so any runner in any environment
can call it.

Usage: python -m salmonid.util.ntfy "message body" [--title "title"] [--priority 4] [--tags x]
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

TOPIC    = "ontario-fish-update"   # project ntfy.sh topic (subscribe with the ntfy app)
NTFY_URL = "https://ntfy.sh"
# Offline safety net: <package>/logs/ntfy_unsent.log (repo-local, resolved from this file).
FALLBACK = Path(__file__).resolve().parents[1] / "logs" / "ntfy_unsent.log"


def send(message: str, title: str | None = None, priority: int = 3,
         tags: str | None = None, attempts: int = 3) -> bool:
    """POST ``message`` to the project ntfy topic, retrying then falling back to disk.

    Args:
        message: Notification body sent as the request payload.
        title: Optional notification title (ntfy ``Title`` header).
        priority: ntfy priority 1-5 (ntfy ``Priority`` header; default 3 = normal).
        tags: Optional comma-separated ntfy tag string (emoji/labels).
        attempts: Number of send tries before giving up (default 3).

    Returns:
        True if a POST succeeded, False if every attempt failed.

    Side effects:
        Performs network I/O and prints a status line. On total failure, appends the
        message to ``FALLBACK`` (creating its parent dir) so nothing is lost offline.
    """
    headers = {}
    if title:
        headers["Title"] = title
    headers["Priority"] = str(priority)
    if tags:
        headers["Tags"] = tags
    data = message.encode("utf-8")
    for i in range(attempts):
        req = urllib.request.Request(f"{NTFY_URL}/{TOPIC}", data=data, headers=headers)
        try:
            urllib.request.urlopen(req, timeout=10)
            print(f"ntfy: sent ({len(message)} chars)")
            return True
        except Exception as exc:
            print(f"ntfy: attempt {i + 1}/{attempts} failed -- {exc}")
            if i < attempts - 1:
                time.sleep(3)   # fixed back-off between retries; no sleep after the last try
    # all attempts failed -> persist locally so nothing is lost during an outage
    try:
        FALLBACK.parent.mkdir(parents=True, exist_ok=True)
        with open(FALLBACK, "a", encoding="utf-8") as f:
            f.write(f"[unsent] title={title!r} prio={priority} :: {message}\n")
        print(f"ntfy: offline -- logged to {FALLBACK}")
    except Exception as exc:
        print(f"ntfy: offline and fallback-log failed -- {exc}")
    return False


def main():
    """CLI entry point: parse ``message``/``--title``/``--priority``/``--tags`` and send once."""
    p = argparse.ArgumentParser()
    p.add_argument("message", help="Notification body")
    p.add_argument("--title",    default=None)
    p.add_argument("--priority", type=int, default=3)
    p.add_argument("--tags",     default=None)
    args = p.parse_args()
    send(args.message, args.title, args.priority, args.tags)


if __name__ == "__main__":
    main()
