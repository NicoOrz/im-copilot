"""Thin wrapper for invoking lark-cli as a subprocess with per-user identity."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

_CLI_PATH = os.environ.get("LARK_CLI_PATH", "/usr/local/bin/lark-cli")


def run_lark_cli(
    args: list[str],
    uat: str = "",
    stdin: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run lark-cli with optional user identity and return parsed JSON output.

    Each call spawns an independent subprocess with its own env, making it
    safe for concurrent multi-user requests.

    Args:
        args: lark-cli arguments (e.g. ["docs", "+create", "--title", "..."]).
        uat: user_access_token for the calling user. If empty, lark-cli falls
             back to its default configured identity.
        stdin: optional string to pipe to the process stdin.
        timeout: subprocess timeout in seconds.

    Returns:
        Parsed JSON dict from stdout, or empty dict on failure.
    """
    env = os.environ.copy()
    if uat:
        env["LARKSUITE_CLI_USER_ACCESS_TOKEN"] = uat
        app_id = os.environ.get("LARK_APP_ID") or os.environ.get("FEISHU_APP_ID", "")
        if app_id:
            env["LARKSUITE_CLI_APP_ID"] = app_id

    cmd = [_CLI_PATH] + args
    logger.debug("lark-cli: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("lark-cli timed out: %s", " ".join(cmd))
        return {}
    except FileNotFoundError:
        logger.error("lark-cli not found at %s", _CLI_PATH)
        return {}

    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        logger.error("lark-cli failed (rc=%d): %s | %s", result.returncode, " ".join(cmd), output)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {}

    output = result.stdout.strip()
    if not output:
        return {}

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        logger.error("lark-cli non-JSON output: %s", output[:200])
        return {}
