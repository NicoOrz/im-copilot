"""FastAPI OAuth callback handler for Feishu user authorization."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from im_copilot.user_token_store import token_store

logger = logging.getLogger(__name__)

router = APIRouter()

_SUCCESS_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>授权成功</title></head>
<body style="font-family:sans-serif;text-align:center;padding:60px">
<h2>✅ 授权成功</h2>
<p>你已成功授权，可以关闭此页面，回到飞书继续使用。</p>
</body></html>"""

_FAIL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>授权失败</title></head>
<body style="font-family:sans-serif;text-align:center;padding:60px">
<h2>❌ 授权失败</h2>
<p>{reason}</p>
</body></html>"""


@router.get("/oauth/callback")
async def oauth_callback(request: Request) -> HTMLResponse:
    code = request.query_params.get("code")
    open_id = request.query_params.get("state")  # we pass open_id as state

    if not code or not open_id:
        return HTMLResponse(_FAIL_HTML.format(reason="缺少 code 或 state 参数"), status_code=400)

    app_id = os.environ.get("FEISHU_APP_ID") or os.environ.get("LARK_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET") or os.environ.get("LARK_APP_SECRET", "")
    redirect_uri = os.environ.get("OAUTH_CALLBACK_URL", "")

    try:
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
        data: dict[str, Any] = resp.json()
    except Exception as e:
        logger.error("OAuth token exchange failed: %s", e)
        return HTMLResponse(_FAIL_HTML.format(reason="网络错误，请重试"), status_code=500)

    if data.get("code", -1) != 0:
        logger.error("OAuth API error: %s", data)
        return HTMLResponse(_FAIL_HTML.format(reason=data.get("error_description", "授权失败")), status_code=400)

    token_store.save(
        open_id=open_id,
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token"),
        expires_in=data.get("expires_in", 7200),
    )
    logger.info("OAuth success for open_id=%s", open_id)
    return HTMLResponse(_SUCCESS_HTML)
