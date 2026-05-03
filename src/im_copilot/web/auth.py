"""WebUI authentication module using Feishu OAuth2."""

from __future__ import annotations

import json
import os
import secrets
import urllib.parse

import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from im_copilot.oauth_scopes import user_oauth_scope_string


_FEISHU_AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
_FEISHU_USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"

_LOGIN_EXEMPT_PATHS = frozenset({"/login", "/logout", "/oauth/callback", "/static"})
_OAUTH_STATE_MAX_AGE_SECONDS = 10 * 60


def _session_secret() -> str:
    return os.environ.get("SESSION_SECRET", "im-copilot-dev-secret-change-me")


def _state_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_session_secret(), salt="im-copilot-oauth-state")


def _sign_web_state(nonce: str) -> str:
    return _state_serializer().dumps({"flow": "web", "nonce": nonce})


def validate_web_state(state: dict) -> bool:
    token = state.get("token")
    nonce = state.get("nonce")
    if not token or not nonce:
        return False
    try:
        payload = _state_serializer().loads(token, max_age=_OAUTH_STATE_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return False
    return payload.get("flow") == "web" and payload.get("nonce") == nonce


def login_redirect_url(request: Request | None = None) -> str:
    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
    callback_url = os.environ.get("OAUTH_CALLBACK_URL", "")
    nonce = secrets.token_urlsafe(16)
    if request is not None:
        request.session["oauth_nonce"] = nonce
    state = json.dumps({"flow": "web", "nonce": nonce, "token": _sign_web_state(nonce)})
    return (
        f"{_FEISHU_AUTHORIZE_URL}"
        f"?app_id={urllib.parse.quote(app_id)}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope={urllib.parse.quote(user_oauth_scope_string())}"
        f"&state={urllib.parse.quote(state)}"
    )


def get_current_user(request: Request) -> dict | None:
    if "session" not in request.scope:
        return None
    open_id = request.session.get("open_id")
    if not open_id:
        return None
    return {
        "open_id": open_id,
        "name": request.session.get("name", ""),
        "avatar_url": request.session.get("avatar_url", ""),
    }


async def require_auth(request: Request):
    user = get_current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=302)
    return user


def is_login_exempt(path: str) -> bool:
    for exempt in _LOGIN_EXEMPT_PATHS:
        if path == exempt or path.startswith(exempt + "/"):
            return True
    return False


async def fetch_user_info(access_token: str) -> dict | None:
    try:
        resp = httpx.get(
            _FEISHU_USER_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        data = resp.json()
    except Exception:
        return None
    if data.get("code", -1) != 0:
        return None
    info = data.get("data", {})
    return {
        "open_id": info.get("open_id", ""),
        "name": info.get("name", ""),
        "avatar_url": info.get("avatar_url", ""),
    }
