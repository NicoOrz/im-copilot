"""Lark (Feishu) document/slide/whiteboard client wrapping ``lark-cli``.

This module provides a high-level interface for creating and editing
Feishu documents, slides, and whiteboards via the ``lark-cli`` command-line
tool.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class LarkDocClient:
    """Client for Feishu document operations via ``lark-cli``.

    Pass ``user_access_token`` to call lark-cli as the user rather than the
    default bot identity.
    """

    def __init__(self, user_access_token: str | None = None) -> None:
        self._cli_path = os.environ.get("LARK_CLI_PATH", "/usr/local/bin/lark-cli")
        self._uat = user_access_token or ""

    # --------------------------------------------------------------------- #
    # Internal helper
    # --------------------------------------------------------------------- #

    def _run_lark_cli(self, args: list[str]) -> dict[str, Any]:
        """Run ``lark-cli`` with *args* and return the parsed JSON response."""
        cmd = [self._cli_path] + args
        logger.debug("Running lark-cli: %s", " ".join(cmd))
        env = os.environ.copy()
        if self._uat:
            env["LARKSUITE_CLI_USER_ACCESS_TOKEN"] = self._uat
            app_id = os.environ.get("LARK_APP_ID") or os.environ.get("FEISHU_APP_ID", "")
            if app_id:
                env["LARKSUITE_CLI_APP_ID"] = app_id
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.error("lark-cli timed out: %s", " ".join(cmd))
            return {}
        except FileNotFoundError:
            logger.error("lark-cli not found at %s", self._cli_path)
            return {}

        if result.returncode != 0:
            # lark-cli often prints JSON error info on stderr/stdout
            output = result.stderr.strip() or result.stdout.strip()
            logger.error(
                "lark-cli failed (rc=%d): %s | output: %s",
                result.returncode,
                " ".join(cmd),
                output,
            )
            # Attempt to parse JSON error output for structured info
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {}

        output = result.stdout.strip()
        if not output:
            logger.warning("lark-cli returned empty stdout: %s", " ".join(cmd))
            return {}

        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse lark-cli JSON output: %s | output: %s",
                exc,
                output,
            )
            return {}

        if not data.get("ok", True):
            err = data.get("error", {})
            logger.error(
                "lark-cli API error: %s | error: %s",
                " ".join(cmd),
                err,
            )

        return data

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def create_doc(self, title: str, folder_token: str | None = None) -> str:
        """Create a new Feishu document and return its document token.

        Args:
            title: Document title.
            folder_token: Optional parent folder token.

        Returns:
            The document token (``document_id``) or an empty string on failure.
        """
        args = [
            "docs",
            "+create",
            "--api-version", "v2",
            "--title", title,
            "--content", " ",  # v2 requires non-empty content
        ]
        if folder_token:
            args.extend(["--parent-token", folder_token])

        logger.info("Creating doc: title=%s folder=%s", title, folder_token)
        resp = self._run_lark_cli(args)
        doc_id = (
            resp.get("data", {})
            .get("document", {})
            .get("document_id", "")
        )
        if doc_id:
            logger.info("Created doc %s", doc_id)
        else:
            logger.error("Failed to create doc: %s", resp)
        return doc_id

    def write_doc(self, doc_token: str, content: str) -> bool:
        """Write Markdown content to an existing Feishu document.

        Args:
            doc_token: The target document token.
            content: Markdown content to write.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """
        args = [
            "docs",
            "+update",
            "--api-version", "v2",
            "--doc", doc_token,
            "--content", content,
            "--command", "overwrite",
        ]
        logger.info("Writing doc %s (content length=%d)", doc_token, len(content))
        resp = self._run_lark_cli(args)
        ok = resp.get("ok", False)
        if ok:
            logger.info("Wrote doc %s", doc_token)
        else:
            logger.error("Failed to write doc %s: %s", doc_token, resp)
        return ok

    def create_slide(self, title: str) -> str:
        """Create a new Feishu Slides presentation and return its token.

        Args:
            title: Presentation title.

        Returns:
            The presentation token or an empty string on failure.
        """
        args = [
            "slides",
            "+create",
            "--title", title,
        ]
        logger.info("Creating slide: title=%s", title)
        resp = self._run_lark_cli(args)
        # The CLI returns the created presentation under data.presentation
        presentation = resp.get("data", {}).get("presentation", {})
        slide_id = presentation.get("presentation_token", "")
        if not slide_id:
            # Fallback: some versions may use obj_token or id directly
            slide_id = presentation.get("obj_token", "") or presentation.get("id", "")
        if slide_id:
            logger.info("Created slide %s", slide_id)
        else:
            logger.error("Failed to create slide: %s", resp)
        return slide_id

    def create_whiteboard(self, title: str) -> str:
        """Create a new Feishu whiteboard and return its whiteboard token.

        Whiteboards are created as wiki nodes of type ``docx`` containing a
        blank whiteboard block.  The whiteboard token is then extracted from
        the document content.

        Args:
            title: Whiteboard title.

        Returns:
            The whiteboard token or an empty string on failure.
        """
        # Step 1: create a docx wiki node
        args_create = [
            "wiki",
            "+node-create",
            "--title", title,
            "--obj-type", "docx",
        ]
        logger.info("Creating whiteboard (docx node): title=%s", title)
        resp_create = self._run_lark_cli(args_create)
        if not resp_create.get("ok", False):
            logger.error("Failed to create whiteboard node: %s", resp_create)
            return ""

        obj_token = resp_create.get("data", {}).get("obj_token", "")
        if not obj_token:
            logger.error("No obj_token in whiteboard creation response: %s", resp_create)
            return ""

        # Step 2: insert a blank whiteboard block
        args_update = [
            "docs",
            "+update",
            "--api-version", "v2",
            "--doc", obj_token,
            "--content", '<whiteboard type="blank"></whiteboard>',
            "--command", "overwrite",
        ]
        resp_update = self._run_lark_cli(args_update)
        if not resp_update.get("ok", False):
            logger.error("Failed to insert whiteboard block: %s", resp_update)
            return ""

        # Step 3: fetch document to extract whiteboard token
        args_fetch = [
            "docs",
            "+fetch",
            "--api-version", "v2",
            "--doc", obj_token,
            "--detail", "full",
        ]
        resp_fetch = self._run_lark_cli(args_fetch)
        if not resp_fetch.get("ok", False):
            logger.error("Failed to fetch whiteboard doc: %s", resp_fetch)
            return ""

        content = (
            resp_fetch.get("data", {})
            .get("document", {})
            .get("content", "")
        )
        # Parse token="..." from the XML content
        import re
        match = re.search(r'token="([^"]+)"', content)
        if match:
            whiteboard_token = match.group(1)
            logger.info("Created whiteboard %s", whiteboard_token)
            return whiteboard_token

        logger.error("Could not extract whiteboard token from content: %s", content)
        return ""

    def get_share_link(self, token: str, type_: str = "doc") -> str:
        """Retrieve a shareable URL for a Feishu document, slide, or whiteboard.

        Args:
            token: The object token (document, slide, or whiteboard token).
            type_: Object type - ``doc`` (default), ``slide``, or ``whiteboard``.

        Returns:
            The shareable URL or an empty string on failure.
        """
        if type_ == "slide":
            # Slides do not have a dedicated fetch command; try to construct
            # the canonical URL from the token.
            url = f"https://www.feishu.cn/slides/{token}"
            logger.info("Slides share link (best-effort): %s", url)
            return url

        if type_ == "whiteboard":
            # Whiteboard tokens are block tokens; we need the parent doc to
            # get a real URL.  Fetch the doc that contains the whiteboard.
            # Since we don't store the parent doc token, try querying the
            # whiteboard directly and fall back to a canonical URL.
            args = [
                "whiteboard",
                "+query",
                "--whiteboard-token", token,
                "--output_as", "raw",
            ]
            resp = self._run_lark_cli(args)
            if resp.get("ok", False):
                # If the whiteboard is inside a doc, the doc URL is the share link
                # lark-cli does not return the parent doc, so construct a URL
                url = f"https://www.feishu.cn/whiteboard/{token}"
                logger.info("Whiteboard share link (best-effort): %s", url)
                return url
            logger.warning("Whiteboard query failed, returning best-effort URL")
            return f"https://www.feishu.cn/whiteboard/{token}"

        # Default: doc
        args = [
            "docs",
            "+fetch",
            "--api-version", "v2",
            "--doc", token,
        ]
        logger.info("Fetching share link for doc %s", token)
        resp = self._run_lark_cli(args)
        doc_data = resp.get("data", {}).get("document", {})
        # v2 fetch does not return URL directly; try to get it from creation
        # response cache or construct it.  The canonical pattern is:
        # https://<tenant>.feishu.cn/docx/<token>
        # We return the URL if available in the response, otherwise construct.
        url = doc_data.get("url", "")
        if not url:
            url = f"https://www.feishu.cn/docx/{token}"
        if resp.get("ok", False):
            logger.info("Doc share link: %s", url)
        else:
            logger.error("Failed to get share link for %s: %s", token, resp)
        return url

    def update_whiteboard(self, whiteboard_token: str, mermaid: str) -> bool:
        """Write Mermaid content to an existing Feishu whiteboard via stdin."""
        cmd = [
            self._cli_path, "whiteboard", "+update", whiteboard_token,
            "--source", "-", "--input_format", "mermaid",
        ]
        env = os.environ.copy()
        if self._uat:
            env["LARKSUITE_CLI_USER_ACCESS_TOKEN"] = self._uat
            app_id = os.environ.get("LARK_APP_ID") or os.environ.get("FEISHU_APP_ID", "")
            if app_id:
                env["LARKSUITE_CLI_APP_ID"] = app_id
        logger.info("Updating whiteboard %s (mermaid length=%d)", whiteboard_token, len(mermaid))
        try:
            result = subprocess.run(
                cmd,
                input=mermaid,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.error("lark-cli whiteboard +update timed out")
            return False
        except FileNotFoundError:
            logger.error("lark-cli not found at %s", self._cli_path)
            return False

        if result.returncode != 0:
            logger.error("whiteboard +update failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
        try:
            data = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError:
            data = {}
        ok = data.get("ok", True)
        if ok:
            logger.info("Updated whiteboard %s", whiteboard_token)
        else:
            logger.error("whiteboard +update API error: %s", data)
        return bool(ok)

    def create_slide_with_content(self, title: str, slides_xml: str) -> str:
        """Create a Feishu Slides presentation with initial slide content.

        Args:
            title: Presentation title.
            slides_xml: Comma-separated ``<slide>`` XML fragments.

        Returns:
            The presentation token or an empty string on failure.
        """
        args = [
            "slides", "+create",
            "--title", title,
            "--slides", f"[{slides_xml}]",
        ]
        logger.info("Creating slide with content: title=%s", title)
        resp = self._run_lark_cli(args)
        presentation = resp.get("data", {}).get("presentation", {})
        slide_id = (
            presentation.get("presentation_token")
            or presentation.get("obj_token")
            or presentation.get("id", "")
        )
        if slide_id:
            logger.info("Created slide with content %s", slide_id)
        else:
            logger.error("Failed to create slide with content: %s", resp)
        return slide_id
