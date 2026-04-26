"""Lark (Feishu) bot wrapper around lark-oapi SDK.

This module provides a high-level interface for sending/receiving messages
and updating interactive cards via the Feishu OpenAPI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import requests

import lark_oapi as lark
from lark_oapi.api.auth.v3 import (
    InternalTenantAccessTokenRequestBuilder,
    InternalTenantAccessTokenRequestBodyBuilder,
)
from lark_oapi.api.im.v1 import (
    CreateMessageRequestBodyBuilder,
    CreateMessageRequestBuilder,
    ReplyMessageRequestBodyBuilder,
    ReplyMessageRequestBuilder,
)
from lark_oapi.core.exception import ObtainAccessTokenException
from lark_oapi.core.token import TokenManager

logger = logging.getLogger(__name__)


class _CallableEventHandler:
    """Wraps a plain callable into an object that satisfies the WS client's
    ``do_without_validation`` interface.
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        self._fn = fn

    def do_without_validation(self, payload: bytes) -> Any:
        data = json.loads(payload.decode("utf-8"))
        return self._fn(data)


class LarkBot:
    """High-level wrapper for Feishu (Lark) bot operations.

    Parameters
    ----------
    app_id:
        Feishu application ID.
    app_secret:
        Feishu application secret.
    encrypt_key:
        Optional encryption key for event payloads.
    verification_token:
        Optional verification token for event validation.
    domain:
        Feishu OpenAPI domain. Defaults to the Chinese Feishu endpoint.
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        encrypt_key: str | None = None,
        verification_token: str | None = None,
        domain: str = lark.FEISHU_DOMAIN,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._encrypt_key = encrypt_key or ""
        self._verification_token = verification_token or ""
        self._domain = domain.rstrip("/")

        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.WARNING)
            .build()
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _get_tenant_access_token(self) -> str:
        """Obtain a tenant access token, using the SDK token manager cache."""
        try:
            return TokenManager.get_self_tenant_token(self._client._config)
        except ObtainAccessTokenException:
            raise
        except Exception as exc:
            logger.exception("Failed to obtain tenant access token")
            raise RuntimeError("Failed to obtain tenant access token") from exc

    @staticmethod
    def _unwrap_response(resp: Any) -> dict[str, Any]:
        """Convert an SDK response object into a plain dict.

        The dict contains ``code``, ``msg``, and ``data`` (when present).
        """
        result: dict[str, Any] = {"code": resp.code, "msg": resp.msg}
        if getattr(resp, "data", None) is not None:
            data = resp.data
            if hasattr(data, "__dict__"):
                result["data"] = {
                    k: v for k, v in data.__dict__.items() if v is not None
                }
            else:
                result["data"] = data
        return result

    # --------------------------------------------------------------------- #
    # Messaging
    # --------------------------------------------------------------------- #

    def send_text(self, chat_id: str, text: str) -> dict[str, Any]:
        """Send a plain text message to a chat.

        Parameters
        ----------
        chat_id:
            The ID of the target chat.
        text:
            The text content to send.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        content = json.dumps({"text": text}, ensure_ascii=False)
        body = (
            CreateMessageRequestBodyBuilder()
            .receive_id(chat_id)
            .msg_type("text")
            .content(content)
            .build()
        )
        req = (
            CreateMessageRequestBuilder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.create(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "send_text failed: code=%s msg=%s chat_id=%s",
                    resp.code,
                    resp.msg,
                    chat_id,
                )
            else:
                logger.info("send_text success: chat_id=%s", chat_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("send_text auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("send_text unexpected error: chat_id=%s", chat_id)
            return {"code": -1, "msg": str(exc), "data": None}

    def reply_text(self, message_id: str, text: str) -> dict[str, Any]:
        """Reply to a specific message with plain text.

        Parameters
        ----------
        message_id:
            The ID of the message to reply to.
        text:
            The text content to send.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        content = json.dumps({"text": text}, ensure_ascii=False)
        body = (
            ReplyMessageRequestBodyBuilder()
            .msg_type("text")
            .content(content)
            .build()
        )
        req = (
            ReplyMessageRequestBuilder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.reply(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "reply_text failed: code=%s msg=%s message_id=%s",
                    resp.code,
                    resp.msg,
                    message_id,
                )
            else:
                logger.info("reply_text success: message_id=%s", message_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("reply_text auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("reply_text unexpected error: message_id=%s", message_id)
            return {"code": -1, "msg": str(exc), "data": None}

    def send_card(self, chat_id: str, card_json: dict[str, Any]) -> dict[str, Any]:
        """Send an interactive card to a chat.

        Parameters
        ----------
        chat_id:
            The ID of the target chat.
        card_json:
            The card JSON payload.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        content = json.dumps(card_json, ensure_ascii=False)
        body = (
            CreateMessageRequestBodyBuilder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(content)
            .build()
        )
        req = (
            CreateMessageRequestBuilder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.create(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "send_card failed: code=%s msg=%s chat_id=%s",
                    resp.code,
                    resp.msg,
                    chat_id,
                )
            else:
                logger.info("send_card success: chat_id=%s", chat_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("send_card auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("send_card unexpected error: chat_id=%s", chat_id)
            return {"code": -1, "msg": str(exc), "data": None}

    def reply_card(self, message_id: str, card_json: dict[str, Any]) -> dict[str, Any]:
        """Reply to a specific message with an interactive card.

        Parameters
        ----------
        message_id:
            The ID of the message to reply to.
        card_json:
            The card JSON payload.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        content = json.dumps(card_json, ensure_ascii=False)
        body = (
            ReplyMessageRequestBodyBuilder()
            .msg_type("interactive")
            .content(content)
            .build()
        )
        req = (
            ReplyMessageRequestBuilder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.reply(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "reply_card failed: code=%s msg=%s message_id=%s",
                    resp.code,
                    resp.msg,
                    message_id,
                )
            else:
                logger.info("reply_card success: message_id=%s", message_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("reply_card auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("reply_card unexpected error: message_id=%s", message_id)
            return {"code": -1, "msg": str(exc), "data": None}

    # --------------------------------------------------------------------- #
    # Card stream updates
    # --------------------------------------------------------------------- #

    def update_card_stream(
        self,
        card_id: str,
        element_id: str,
        content: str,
        sequence: int,
    ) -> dict[str, Any]:
        """Stream-update card content via the cardkit API.

        This calls the Feishu OpenAPI directly using ``requests`` because the
        cardkit stream endpoint may not be available in the SDK.

        Parameters
        ----------
        card_id:
            The card identifier.
        element_id:
            The element within the card to update.
        content:
            The new content string.
        sequence:
            Monotonic sequence number for the stream.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        token = self._get_tenant_access_token()
        url = (
            f"{self._domain}/open-apis/cardkit/v1/cards/"
            f"{card_id}/elements/{element_id}/content"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "content": content,
            "sequence": sequence,
        }

        try:
            response = requests.put(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") != 0:
                logger.error(
                    "update_card_stream failed: code=%s msg=%s card_id=%s element_id=%s",
                    result.get("code"),
                    result.get("msg"),
                    card_id,
                    element_id,
                )
            else:
                logger.info(
                    "update_card_stream success: card_id=%s element_id=%s sequence=%s",
                    card_id,
                    element_id,
                    sequence,
                )
            return result
        except requests.HTTPError as exc:
            logger.error(
                "update_card_stream HTTP error: %s card_id=%s element_id=%s",
                exc,
                card_id,
                element_id,
            )
            return {"code": exc.response.status_code, "msg": str(exc), "data": None}
        except requests.RequestException as exc:
            logger.error(
                "update_card_stream request error: %s card_id=%s element_id=%s",
                exc,
                card_id,
                element_id,
            )
            return {"code": -1, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception(
                "update_card_stream unexpected error: card_id=%s element_id=%s",
                card_id,
                element_id,
            )
            return {"code": -1, "msg": str(exc), "data": None}

    # --------------------------------------------------------------------- #
    # WebSocket
    # --------------------------------------------------------------------- #

    def start_ws(self, event_handler: Callable[[dict[str, Any]], Any]) -> None:
        """Start the WebSocket client to receive real-time events.

        Parameters
        ----------
        event_handler:
            A callable that receives event payloads as dicts.
            This method blocks while the connection is alive.
        """
        handler = _CallableEventHandler(event_handler)
        ws_client = lark.ws.Client(
            app_id=self._app_id,
            app_secret=self._app_secret,
            event_handler=handler,
            domain=self._domain,
            auto_reconnect=True,
        )
        logger.info("Starting Lark WebSocket client for app_id=%s", self._app_id)
        ws_client.start()
