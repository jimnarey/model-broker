from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable

import httpx
import pytest

from model_broker.events import (
    LOGGER,
    REQUEST,
    AuthenticationOff,
    Event,
    JsonFormatter,
    ModelLoaded,
    RequestAdmitted,
    RequestContext,
    RouterProblemsFound,
    configure_logging,
    emit,
)
from model_broker.instrumentation import requested_model


class Lines(logging.Handler):
    """Collect formatted lines instead of writing them."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def formatted(event: Event, context: RequestContext | None = None) -> dict[str, object]:
    """Emit ``event`` (inside ``context`` if given) and return the JSON line it produces."""
    handler = Lines()
    LOGGER.addHandler(handler)
    token = REQUEST.set(context)
    try:
        emit(event)
    finally:
        REQUEST.reset(token)
        LOGGER.removeHandler(handler)
    return json.loads(handler.lines[0])


@pytest.fixture(autouse=True)
def debug_level() -> None:
    configure_logging("DEBUG")


def test_event_names_are_the_class_names_in_snake_case() -> None:
    assert ModelLoaded.name() == "model_loaded"
    assert RouterProblemsFound.name() == "router_problems_found"


def test_an_event_is_one_json_line_of_its_facts() -> None:
    """The line has the time, level and event name, then the event's fields.

    Inside a request, the request ID and client are added, so that everything a request
    caused can be found by its ID.
    """
    line = formatted(ModelLoaded("m--cuda0", 41.2), RequestContext("ab12cd34", "vscode"))

    assert line.pop("time")
    assert line == {
        "level": "info",
        "event": "model_loaded",
        "request_id": "ab12cd34",
        "client": "vscode",
        "model": "m--cuda0",
        "seconds": 41.2,
    }


def test_an_event_outside_a_request_has_no_request_fields() -> None:
    line = formatted(AuthenticationOff())
    assert (line["level"], line["event"]) == ("warning", "authentication_off")
    assert "request_id" not in line


def test_other_records_keep_their_message_and_name_their_logger() -> None:
    """Records that are not events, such as a library's, are still valid JSON lines."""
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, "plain message", None, None)
    line = json.loads(JsonFormatter().format(record))
    assert (line["event"], line["logger"]) == ("plain message", "httpx")


def test_exceptions_are_included() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, exc_info)
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exception"]


def test_events_below_the_level_are_not_emitted(events: Callable[[], list[Event]]) -> None:
    """At INFO, a DEBUG event such as an admission is skipped before it is formatted."""
    configure_logging("INFO")
    emit(RequestAdmitted("m--cuda0", 0.0))
    emit(ModelLoaded("m--cuda0", 1.0))
    assert events() == [ModelLoaded("m--cuda0", 1.0)]


def test_configuring_again_replaces_the_handler() -> None:
    """Each app creation configures logging; lines must not be duplicated."""
    configure_logging("INFO")
    configure_logging("DEBUG")
    names = [handler.get_name() for handler in LOGGER.handlers]
    assert names.count("model_broker_json") == 1
    assert LOGGER.level == logging.DEBUG


@pytest.mark.parametrize(
    ("request_", "model"),
    [
        (httpx.Request("GET", "http://r/slots", params={"model": "a--cpu"}), "a--cpu"),
        (httpx.Request("POST", "http://r/models/load", json={"model": "b--cuda0"}), "b--cuda0"),
        (
            httpx.Request(
                "POST",
                "http://r/v1/chat/completions",
                json={"model": "c--cuda1", "messages": [{"role": "user", "content": "secret"}]},
            ),
            "c--cuda1",
        ),
        (httpx.Request("GET", "http://r/props"), None),
        (httpx.Request("POST", "http://r/x", content=b"not json"), None),
    ],
    ids=["query-parameter", "control-body", "chat-body", "no-model", "not-json"],
)
def test_router_calls_report_only_the_model(request_: httpx.Request, model: str | None) -> None:
    """A router call's event names its model, found in the query or the JSON body.

    Only the model is taken from a body: the chat request's messages never reach the log.
    """
    assert requested_model(request_) == model
