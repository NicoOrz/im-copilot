"""FastAPI OAuth callback handler for Feishu user authorization."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from im_copilot.user_token_store import token_store

logger = logging.getLogger(__name__)

router = APIRouter()

_SUCCESS_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>授权成功</title></head>
<body style="font-family:sans-serif;text-align:center;padding:60px">
<h2>授权成功</h2>
<p>你已成功授权，可以关闭此页面，回到飞书继续使用。</p>
</body></html>"""

_FAIL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>授权失败</title></head>
<body style="font-family:sans-serif;text-align:center;padding:60px">
<h2>授权失败</h2>
<p>{reason}</p>
</body></html>"""


def _exchange_code(code: str, redirect_uri: str) -> dict[str, Any]:
    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET") or os.environ.get("LARK_APP_SECRET", "")
    resp = httpx.post(
        "https://open.feishu.cn/open-apis/authen/v2/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": app_id,
            "client_secret": app_secret,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=10,
    )
    return resp.json()


@router.get("/oauth/callback", response_model=None)
async def oauth_callback(request: Request) -> HTMLResponse | RedirectResponse:
    code = request.query_params.get("code")
    state_raw = request.query_params.get("state", "")

    if not code:
        return HTMLResponse(_FAIL_HTML.format(reason="缺少 code 参数"), status_code=400)

    # Detect flow type from state parameter
    flow = "bot"
    open_id = state_raw
    try:
        state_data = json.loads(state_raw)
        if isinstance(state_data, dict) and state_data.get("flow") == "web":
            flow = "web"
            if state_data.get("nonce") != request.session.get("oauth_nonce"):
                return RedirectResponse("/login?error=state", status_code=302)
    except (json.JSONDecodeError, TypeError):
        pass

    redirect_uri = os.environ.get("OAUTH_CALLBACK_URL", "")

    try:
        data = _exchange_code(code, redirect_uri)
    except Exception as e:
        logger.error("OAuth token exchange failed: %s", e)
        if flow == "web":
            return RedirectResponse("/login?error=network", status_code=302)
        return HTMLResponse(_FAIL_HTML.format(reason="网络错误，请重试"), status_code=500)

    if data.get("code", -1) != 0:
        logger.error("OAuth API error: %s", data)
        if flow == "web":
            return RedirectResponse("/login?error=auth", status_code=302)
        return HTMLResponse(_FAIL_HTML.format(reason=data.get("error_description", "授权失败")), status_code=400)

    access_token = data["access_token"]

    if flow == "web":
        from im_copilot.web.auth import fetch_user_info

        user_info = await fetch_user_info(access_token)
        if not user_info or not user_info.get("open_id"):
            return RedirectResponse("/login?error=userinfo", status_code=302)

        open_id = user_info["open_id"]
        token_store.save(
            open_id=open_id,
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in", 7200),
        )

        request.session["open_id"] = open_id
        request.session["name"] = user_info.get("name", "")
        request.session["avatar_url"] = user_info.get("avatar_url", "")
        request.session.pop("oauth_nonce", None)
        logger.info("WebUI OAuth success for open_id=%s name=%s", open_id, user_info.get("name"))
        return RedirectResponse("/", status_code=302)

    # Bot flow: open_id is passed as bare state string
    if not open_id:
        return HTMLResponse(_FAIL_HTML.format(reason="缺少 state 参数"), status_code=400)

    token_store.save(
        open_id=open_id,
        access_token=access_token,
        refresh_token=data.get("refresh_token"),
        expires_in=data.get("expires_in", 7200),
    )
    logger.info("Bot OAuth success for open_id=%s", open_id)
    return HTMLResponse(_SUCCESS_HTML)
