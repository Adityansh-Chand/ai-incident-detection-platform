"""Track which services are currently in an incident.

Point scoring answers "is this minute anomalous". Another service asking "is
checkout broken right now?" needs something the detector did not previously
keep: state across minutes.

The rule here is deliberately simple and stated rather than tuned: a service is
**active** when it has produced at least `MIN_ANOMALIES` anomalous scores inside
the trailing `WINDOW_SECONDS`, and it clears once that stops being true. Single
anomalous minutes do not open an incident, because single anomalous minutes are
mostly noise -- that is the same reasoning behind measuring episode recall rather
than point recall.

This is in-memory and per-process on purpose. It is a demonstration of the
integration surface, not a durable incident store; a real deployment would put
this in shared state so every replica agrees. Said plainly in the README.
"""
import os
import threading
import time
from collections import defaultdict, deque

WINDOW_SECONDS = float(os.getenv("INCIDENT_WINDOW_SECONDS", "900"))
MIN_ANOMALIES = int(os.getenv("INCIDENT_MIN_ANOMALIES", "3"))
MAX_TRACKED_PER_SERVICE = 500

_lock = threading.Lock()
_events = defaultdict(lambda: deque(maxlen=MAX_TRACKED_PER_SERVICE))


def _now():
    return time.time()


def _prune(service, now):
    window = _events[service]
    while window and now - window[0]["at"] > WINDOW_SECONDS:
        window.popleft()


def record(service, score, is_anomaly, severity_hint=None):
    """Record one scored event. Only anomalies are retained."""
    if not is_anomaly:
        return
    now = _now()
    with _lock:
        _prune(service, now)
        _events[service].append(
            {"at": now, "score": float(score), "severity": severity_hint}
        )


def active_incident(service):
    """Return the incident for one service, or None."""
    now = _now()
    with _lock:
        _prune(service, now)
        window = list(_events[service])

    if len(window) < MIN_ANOMALIES:
        return None

    scores = [event["score"] for event in window]
    started = min(event["at"] for event in window)
    return {
        "service": service,
        "anomaly_count": len(window),
        "peak_score": round(max(scores), 6),
        "mean_score": round(sum(scores) / len(scores), 6),
        "started_at_epoch": round(started, 3),
        "duration_seconds": round(now - started, 1),
        "window_seconds": WINDOW_SECONDS,
        "confidence": "heuristic: count of anomalous minutes in a trailing window",
    }


def active_incidents():
    with _lock:
        services = list(_events)
    found = [active_incident(service) for service in services]
    return [incident for incident in found if incident]


def reset():
    """Test hook."""
    with _lock:
        _events.clear()
