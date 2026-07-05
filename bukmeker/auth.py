"""Optional password gate for the Streamlit dashboard.

Design: the dashboard has no user accounts or persistence, so "auth" here
means one shared password protecting the whole app (not per-user login) --
the smallest change that stops "anyone on the network can open the bankroll/
coupon tools" from being true by default, without pretending to be a full
multi-tenant auth system.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

_SESSION_KEY = "_bukmeker_authenticated"
_ENV_VAR = "BUKMEKER_DASHBOARD_PASSWORD"


def require_password() -> bool:
    """Returns True if the caller may proceed to render the app.

    If `BUKMEKER_DASHBOARD_PASSWORD` is not set, the dashboard runs
    unprotected with a visible warning (local/dev use only) -- this is a
    deliberate default so the app still works out of the box for `bukmeker
    dashboard` without requiring extra setup.
    """
    password = os.environ.get(_ENV_VAR)
    if not password:
        st.warning(
            f"Дашборд запущен без пароля ({_ENV_VAR} не задан) — "
            "подходит только для локального использования на своей машине."
        )
        return True

    if st.session_state.get(_SESSION_KEY):
        return True

    st.title("Bukmeker — вход")
    entered = st.text_input("Пароль", type="password", key="_bukmeker_password_input")
    if st.button("Войти", key="_bukmeker_login_button"):
        # constant-time comparison: avoids leaking password length/prefix via response timing
        if hmac.compare_digest(entered, password):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
        else:
            st.error("Неверный пароль.")
    return False
