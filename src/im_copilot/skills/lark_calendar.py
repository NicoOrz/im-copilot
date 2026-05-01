from __future__ import annotations

import logging
from typing import Any

from im_copilot.lark_cli import run_lark_cli

logger = logging.getLogger(__name__)


def create_calendar_event(
    *,
    summary: str,
    start: str,
    end: str,
    attendee_ids: list[str] | None = None,
    description: str = "",
    user_access_token: str = "",
) -> dict[str, Any]:
    if not user_access_token:
        return {"status": "error", "error": "missing_user_access_token"}
    if not start or not end:
        return {"status": "error", "error": "missing_event_time"}

    args = [
        "calendar",
        "+create",
        "--summary",
        summary or "会议",
        "--start",
        start,
        "--end",
        end,
        "--as",
        "user",
    ]
    if description:
        args.extend(["--description", description])
    if attendee_ids:
        args.extend(["--attendee-ids", ",".join(_dedupe(attendee_ids))])

    logger.info(
        "calendar_create start summary=%r start=%s end=%s attendee_count=%s",
        summary,
        start,
        end,
        len(attendee_ids or []),
    )
    resp = run_lark_cli(args, uat=user_access_token)
    if resp.get("ok") is False or resp.get("code") not in (None, 0):
        logger.warning("calendar_create failed response=%s", resp)
        return {"status": "error", "error": resp.get("msg") or resp.get("error") or "calendar_create_failed", "raw": resp}

    data = resp.get("data") or {}
    event = data.get("event") or data.get("calendar_event") or data
    token = str(event.get("event_id") or event.get("id") or data.get("event_id") or "")
    url = str(event.get("url") or event.get("app_link") or data.get("url") or "")
    return {
        "status": "created",
        "token": token,
        "url": url,
        "raw": resp,
    }


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item)
    return result
