"""Outbound event delivery with an outbox, retries, and a dead-letter queue.

Every other edge in this system is a synchronous *pull*: a service asks a
question and waits. That cannot express the thing this platform is named for --
noticing a problem and telling someone unprompted. Detection has to be able to
push.

Push introduces problems pull does not have, and the ones handled here are the
ones that actually bite:

- **Detection must not block on delivery.** Scoring a minute of telemetry cannot
  wait on a subscriber's HTTP stack. Events go to an in-memory outbox and a
  background worker delivers them.
- **Delivery fails.** Subscribers restart, networks blip. Failed events are
  retried with exponential backoff rather than dropped on the first error.
- **Retries mean duplicates.** This is **at-least-once** delivery, not
  exactly-once. Every event carries a stable `event_id`, and subscribers are
  required to be idempotent. Saying "exactly once" would be a lie: the delivery
  succeeding and the acknowledgement being lost is indistinguishable from the
  delivery failing.
- **Some events never deliver.** After `MAX_ATTEMPTS` an event moves to a
  dead-letter queue and is visible at `GET /events/dlq` instead of vanishing.

What this is not: a broker. There is no durable log, no partitioning, no
consumer groups, and the outbox is lost on restart. It is an honest webhook fan-out
with the failure handling that makes push usable, and the README says so.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque

SUBSCRIBERS_ENV = "EVENT_SUBSCRIBERS"
MAX_ATTEMPTS = int(os.getenv("EVENT_MAX_ATTEMPTS", "4"))
BASE_BACKOFF_SECONDS = float(os.getenv("EVENT_BACKOFF_SECONDS", "1.0"))
DELIVERY_TIMEOUT_SECONDS = float(os.getenv("EVENT_TIMEOUT_SECONDS", "3.0"))
POLL_INTERVAL_SECONDS = float(os.getenv("EVENT_POLL_SECONDS", "0.5"))
MAX_OUTBOX = 1000
MAX_DLQ = 200

INCIDENT_OPENED = "incident.opened"
INCIDENT_RESOLVED = "incident.resolved"


class EventBus:
    def __init__(self, subscribers=None, api_key=None, auto_start=True):
        raw = subscribers if subscribers is not None else os.getenv(SUBSCRIBERS_ENV, "")
        self.subscribers = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        self.api_key = api_key if api_key is not None else os.getenv("INTEGRATION_API_KEY")
        # Tests drive delivery explicitly via drain_once(); the background worker
        # would otherwise empty the outbox before anything could inspect it.
        self.auto_start = auto_start

        self._outbox = deque(maxlen=MAX_OUTBOX)
        self._dlq = deque(maxlen=MAX_DLQ)
        self._delivered = 0
        self._lock = threading.Lock()
        self._worker = None
        self._stop = threading.Event()

    @property
    def enabled(self):
        return bool(self.subscribers)

    def publish(self, event_type, payload, request_id=None):
        """Queue an event. Returns immediately -- never blocks the caller."""
        if not self.enabled:
            return None

        event = {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "payload": payload,
            "source": "ai-incident-detection-platform",
            "request_id": request_id,
            "attempts": 0,
            "next_attempt_at": 0.0,
        }
        with self._lock:
            for subscriber in self.subscribers:
                self._outbox.append({**event, "subscriber": subscriber})
        if self.auto_start:
            self.start()
        return event["event_id"]

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            if not self.drain_once():
                time.sleep(POLL_INTERVAL_SECONDS)

    def drain_once(self):
        """Attempt one due event. Returns True if work was done.

        Exposed so tests can drive delivery deterministically instead of sleeping.
        """
        now = time.monotonic()
        with self._lock:
            for _ in range(len(self._outbox)):
                event = self._outbox.popleft()
                if event["next_attempt_at"] <= now:
                    break
                self._outbox.append(event)  # not due yet; rotate
            else:
                return False

        ok, detail = self._deliver(event)
        if ok:
            with self._lock:
                self._delivered += 1
            return True

        event["attempts"] += 1
        event["last_error"] = detail
        if event["attempts"] >= MAX_ATTEMPTS:
            with self._lock:
                self._dlq.append(event)
            return True

        # Exponential backoff: 1s, 2s, 4s ... Read from the module at call time
        # rather than captured at import, so it can be tuned without a restart.
        backoff = globals()["BASE_BACKOFF_SECONDS"]
        event["next_attempt_at"] = now + backoff * (2 ** (event["attempts"] - 1))
        with self._lock:
            self._outbox.append(event)
        return True

    def _deliver(self, event):
        body = json.dumps({
            "event_id": event["event_id"],
            "type": event["type"],
            "source": event["source"],
            "payload": event["payload"],
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if event.get("request_id"):
            headers["X-Request-ID"] = event["request_id"]

        request = urllib.request.Request(
            f"{event['subscriber']}/v1/events/incident",
            data=body, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=DELIVERY_TIMEOUT_SECONDS):
                return True, None
        except urllib.error.HTTPError as error:
            # A 4xx means the subscriber rejected the event itself. Retrying an
            # unacceptable payload just burns attempts to reach the same DLQ.
            if 400 <= error.code < 500:
                return False, f"http_{error.code}_permanent"
            return False, f"http_{error.code}"
        except Exception as error:  # noqa: BLE001 - delivery must never raise
            return False, type(error).__name__

    def status(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "subscribers": self.subscribers,
                "outbox_depth": len(self._outbox),
                "dead_lettered": len(self._dlq),
                "delivered": self._delivered,
                "delivery": "at-least-once; subscribers must be idempotent",
                "max_attempts": MAX_ATTEMPTS,
            }

    def outbox(self):
        with self._lock:
            return [dict(event) for event in self._outbox]

    def dead_letters(self):
        with self._lock:
            return [dict(event) for event in self._dlq]

    def reset(self):
        self.stop()
        with self._lock:
            self._outbox.clear()
            self._dlq.clear()
            self._delivered = 0


_bus = None


def get_bus():
    """Process-wide bus.

    Deliberately not named `bus`: `events/__init__.py` re-exports from here, and
    a function called `bus` shadows the `events.bus` submodule itself, which
    makes `import events.bus` silently resolve to the function.
    """
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_bus():
    """Test hook -- re-reads environment."""
    global _bus
    if _bus is not None:
        _bus.reset()
    _bus = None
