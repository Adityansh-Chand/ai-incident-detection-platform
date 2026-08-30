"""Outbound event delivery: retries, dead-lettering, and transition semantics.

The failure paths are the point. A bus whose failures vanish is worse than no
bus, because it looks like it is working.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from events.bus import INCIDENT_OPENED, INCIDENT_RESOLVED, MAX_ATTEMPTS, EventBus
from monitoring.incident_state import MIN_ANOMALIES, record, reset


class _Sub(BaseHTTPRequestHandler):
    received = []
    status_code = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append({"body": body, "headers": dict(self.headers)})
        if type(self).status_code != 200:
            self.send_error(type(self).status_code)
            return
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def subscriber():
    _Sub.received = []
    _Sub.status_code = 200
    server = HTTPServer(("127.0.0.1", 0), _Sub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def clean_state():
    reset()
    yield
    reset()


# --- transitions, not every anomalous minute ----------------------------------

def test_incident_opens_once_not_every_anomalous_minute():
    """Publishing per anomalous minute would spam subscribers with the same news."""
    transitions = [record("checkout", 0.9, True) for _ in range(MIN_ANOMALIES + 3)]
    assert transitions.count("opened") == 1
    assert transitions[MIN_ANOMALIES - 1] == "opened"


def test_healthy_traffic_produces_no_transition():
    assert [record("search", 0.1, False) for _ in range(5)] == [None] * 5


def test_bus_disabled_when_no_subscribers_configured():
    bus = EventBus(subscribers="")
    assert bus.enabled is False
    assert bus.publish(INCIDENT_OPENED, {"service": "checkout"}) is None


# --- delivery -----------------------------------------------------------------

def test_event_is_delivered_to_a_subscriber(subscriber):
    bus = EventBus(subscribers=subscriber, auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "checkout"}, request_id="req-1")

    assert bus.drain_once() is True
    assert len(_Sub.received) == 1
    body = _Sub.received[0]["body"]
    assert body["type"] == INCIDENT_OPENED
    assert body["payload"]["service"] == "checkout"
    assert body["event_id"]


def test_request_id_propagates_through_the_bus(subscriber):
    bus = EventBus(subscribers=subscriber, auto_start=False)
    bus.publish(INCIDENT_RESOLVED, {"service": "checkout"}, request_id="req-trace-9")
    bus.drain_once()

    forwarded = [
        value for entry in _Sub.received
        for key, value in entry["headers"].items() if key.lower() == "x-request-id"
    ]
    assert "req-trace-9" in forwarded


def test_publish_does_not_block_the_caller():
    """Detection must never wait on a subscriber's HTTP stack."""
    bus = EventBus(subscribers="http://127.0.0.1:9", auto_start=False)
    bus.stop()
    event_id = bus.publish(INCIDENT_OPENED, {"service": "checkout"})
    assert event_id  # returned immediately, before any delivery attempt
    assert bus.status()["outbox_depth"] == 1


def test_fanout_to_multiple_subscribers(subscriber):
    bus = EventBus(subscribers=f"{subscriber},{subscriber}", auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "checkout"})
    assert bus.status()["outbox_depth"] == 2


# --- failure handling ---------------------------------------------------------

def test_failed_delivery_is_retried_then_dead_lettered():
    bus = EventBus(subscribers="http://127.0.0.1:9", auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "checkout"})

    import events.bus as module
    original = module.BASE_BACKOFF_SECONDS
    module.BASE_BACKOFF_SECONDS = 0.0  # no waiting between attempts in the test
    try:
        for _ in range(MAX_ATTEMPTS):
            bus.drain_once()
    finally:
        module.BASE_BACKOFF_SECONDS = original

    status = bus.status()
    assert status["dead_lettered"] == 1
    assert status["outbox_depth"] == 0
    assert bus.dead_letters()[0]["attempts"] == MAX_ATTEMPTS


def test_dead_letters_are_inspectable_not_silently_dropped():
    bus = EventBus(subscribers="http://127.0.0.1:9", auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "payments"})

    import events.bus as module
    original = module.BASE_BACKOFF_SECONDS
    module.BASE_BACKOFF_SECONDS = 0.0
    try:
        for _ in range(MAX_ATTEMPTS):
            bus.drain_once()
    finally:
        module.BASE_BACKOFF_SECONDS = original

    letters = bus.dead_letters()
    assert letters
    assert letters[0]["payload"]["service"] == "payments"
    assert letters[0]["last_error"]


def test_subscriber_error_does_not_raise_into_detection(subscriber):
    _Sub.status_code = 500
    bus = EventBus(subscribers=subscriber, auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "checkout"})
    bus.drain_once()  # must not raise
    assert bus.status()["delivered"] == 0


def test_a_rejected_payload_is_not_retried_forever(subscriber):
    """A 4xx means the subscriber rejected the event; retrying burns attempts."""
    _Sub.status_code = 400
    bus = EventBus(subscribers=subscriber, auto_start=False)
    bus.publish(INCIDENT_OPENED, {"service": "checkout"})
    bus.drain_once()
    assert bus.outbox()[0]["last_error"].endswith("_permanent")
