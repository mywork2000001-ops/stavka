"""Headless checks for the dashboard password gate. Verifies both the
unprotected default (no env var set) and the protected path (wrong password
blocks, correct password unlocks) without a real browser."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_APP_PATH = str(Path(__file__).resolve().parent.parent / "bukmeker" / "dashboard.py")
_ENV_VAR = "BUKMEKER_DASHBOARD_PASSWORD"


def test_dashboard_runs_unprotected_with_warning_when_no_password_set(monkeypatch):
    monkeypatch.delenv(_ENV_VAR, raising=False)
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert [t.value for t in at.title] == ["Справка: как пользоваться"]
    assert any("без пароля" in w.value for w in at.warning)


def test_dashboard_blocks_content_until_correct_password_entered(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "s3cret")
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    # gated: shows the login screen's own title, not any dashboard page title
    assert [t.value for t in at.title] == ["Bukmeker — вход"]
    assert len(at.text_input) == 1

    # wrong password -> stays locked, shows an error
    at.text_input[0].set_value("nope").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert [t.value for t in at.title] == ["Bukmeker — вход"]
    assert len(at.error) == 1

    # correct password -> unlocks, lands on the default (help) page
    at.text_input[0].set_value("s3cret").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    assert [t.value for t in at.title] == ["Справка: как пользоваться"]


def test_dashboard_password_session_persists_across_reruns(monkeypatch):
    monkeypatch.setenv(_ENV_VAR, "s3cret")
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    at.text_input[0].set_value("s3cret").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert [t.value for t in at.title] == ["Справка: как пользоваться"]

    # a later rerun (e.g. clicking a widget on some page) must not re-lock
    at.run(timeout=30)
    assert [t.value for t in at.title] == ["Справка: как пользоваться"]
