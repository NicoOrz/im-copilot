"""WebUI authentication module using Feishu OAuth2."""

from __future__ import annotations

import json
import os
import secrets
import urllib.parse

import httpx
from fastapi import Request
from fastapi.responses import RedirectResponse


_FEISHU_AUTHORIZE_URL = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
_FEISHU_USER_INFO_URL = "https://open.feishu.cn/open-apis/authen/v1/user_info"

_WEB_OAUTH_SCOPES = " ".join([
    "offline_access",
    "contact:user.base:readonly",
    "docx:document",
    "drive:drive",
    "slides:presentation:read",
    "slides:presentation:create",
    "slides:presentation:write_only",
    "slides:presentation:update",
    "wiki:wiki",
])

_LOGIN_EXEMPT_PATHS = frozenset({"/login", "/logout", "/oauth/callback", "/static"})


def login_redirect_url(request: Request | None = None) -> str:
    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
    callback_url = os.environ.get("OAUTH_CALLBACK_URL", "")
    nonce = secrets.token_urlsafe(16)
    if request is not None:
        request.session["oauth_nonce"] = nonce
    state = json.dumps({"flow": "web", "nonce": nonce})
    return (
        f"{_FEISHU_AUTHORIZE_URL}"
        f"?app_id={urllib.parse.quote(app_id)}"
        f"&redirect_uri={urllib.parse.quote(callback_url)}"
        f"&response_type=code"
        f"&scope={urllib.parse.quote(_WEB_OAUTH_SCOPES)}"
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
