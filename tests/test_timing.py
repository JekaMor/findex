import logging

import pytest

from findex.timing import timed


@timed
def add(a, b):
    """Add two numbers."""
    return a + b


@timed
def boom():
    raise RuntimeError("boom")


def test_wraps_keeps_name_and_doc():
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."
    assert add.__wrapped__(1, 2) == 3


def test_logs_elapsed_time(caplog):
    with caplog.at_level(logging.INFO, logger="findex.timing"):
        assert add(1, 2) == 3

    (record,) = caplog.records
    assert record.getMessage().startswith("add took ")
    assert record.getMessage().endswith(" ms")


def test_logs_even_when_the_call_raises(caplog):
    with (
        caplog.at_level(logging.INFO, logger="findex.timing"),
        pytest.raises(RuntimeError),
    ):
        boom()

    assert caplog.records[0].getMessage().startswith("boom took")
