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
    CreateMessageReactionRequestBodyBuilder,
    CreateMessageReactionRequestBuilder,
    CreateMessageRequestBodyBuilder,
    CreateMessageRequestBuilder,
    EmojiBuilder,
    PatchMessageRequestBodyBuilder,
    PatchMessageRequestBuilder,
    ReplyMessageRequestBodyBuilder,
    ReplyMessageRequestBuilder,
)
from lark_oapi.core.exception import ObtainAccessTokenException
from lark_oapi.core.token import TokenManager

logger = logging.getLogger(__name__)


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
        debug: bool = False,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._encrypt_key = encrypt_key or ""
        self._verification_token = verification_token or ""
        self._domain = domain.rstrip("/")
        self._debug = debug

        self._client = (
            lark.Client.builder()
            .app_id(app_id)
            .app_secret(app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.DEBUG if debug else lark.LogLevel.WARNING)
            .build()
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _get_tenant_access_token(self) -> str:
        """Obtain a tenant access token, using the SDK token manager cache."""
        try:
            logger.debug("Obtaining tenant access token")
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
        logger.debug("send_text start: chat_id=%s text_len=%s", chat_id, len(text))
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

    def send_text_to_open_id(self, open_id: str, text: str) -> dict[str, Any]:
        """Send a plain text message to a user by open_id."""
        logger.debug("send_text_to_open_id start: open_id=%s text_len=%s", open_id, len(text))
        content = json.dumps({"text": text}, ensure_ascii=False)
        body = (
            CreateMessageRequestBodyBuilder()
            .receive_id(open_id)
            .msg_type("text")
            .content(content)
            .build()
        )
        req = (
            CreateMessageRequestBuilder()
            .receive_id_type("open_id")
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.create(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "send_text_to_open_id failed: code=%s msg=%s open_id=%s",
                    resp.code,
                    resp.msg,
                    open_id,
                )
            else:
                logger.info("send_text_to_open_id success: open_id=%s", open_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("send_text_to_open_id auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("send_text_to_open_id unexpected error: open_id=%s", open_id)
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
        logger.debug("reply_text start: message_id=%s text_len=%s", message_id, len(text))
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

    def add_reaction(self, message_id: str, emoji_type: str = "OK") -> dict[str, Any]:
        """Add an emoji reaction to a message."""
        logger.debug("add_reaction start: message_id=%s emoji_type=%s", message_id, emoji_type)
        body = (
            CreateMessageReactionRequestBodyBuilder()
            .reaction_type(EmojiBuilder().emoji_type(emoji_type).build())
            .build()
        )
        req = (
            CreateMessageReactionRequestBuilder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message_reaction.create(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "add_reaction failed: code=%s msg=%s message_id=%s emoji_type=%s",
                    resp.code,
                    resp.msg,
                    message_id,
                    emoji_type,
                )
            else:
                logger.info("add_reaction success: message_id=%s emoji_type=%s", message_id, emoji_type)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("add_reaction auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("add_reaction unexpected error: message_id=%s", message_id)
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
        logger.debug("send_card start: chat_id=%s card_keys=%s", chat_id, list(card_json.keys()))
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
        logger.debug("reply_card start: message_id=%s card_keys=%s", message_id, list(card_json.keys()))
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
    # Card entity & stream updates (for LLM streaming)
    # --------------------------------------------------------------------- #

    def create_card_entity(self, card_json: dict[str, Any]) -> dict[str, Any]:
        """Create a card entity and return its card_id.

        This is required for streaming text updates via cardkit API.
        """
        logger.debug("create_card_entity start: card_keys=%s", list(card_json.keys()))
        token = self._get_tenant_access_token()
        url = f"{self._domain}/open-apis/cardkit/v1/cards"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "type": "card_json",
            "data": json.dumps(card_json, ensure_ascii=False),
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if result.get("code") != 0:
                logger.error(
                    "create_card_entity failed: code=%s msg=%s",
                    result.get("code"),
                    result.get("msg"),
                )
            else:
                logger.info("create_card_entity success")
            return result
        except requests.HTTPError as exc:
            logger.error("create_card_entity HTTP error: %s", exc)
            return {"code": exc.response.status_code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("create_card_entity unexpected error")
            return {"code": -1, "msg": str(exc), "data": None}

    def send_card_entity(self, chat_id: str, card_id: str) -> dict[str, Any]:
        """Send a card entity (created via create_card_entity) to a chat."""
        logger.debug("send_card_entity start: chat_id=%s card_id=%s", chat_id, card_id)
        content = json.dumps(
            {"type": "card", "data": {"card_id": card_id}},
            ensure_ascii=False,
        )
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
                    "send_card_entity failed: code=%s msg=%s chat_id=%s",
                    resp.code,
                    resp.msg,
                    chat_id,
                )
            else:
                logger.info("send_card_entity success: chat_id=%s", chat_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("send_card_entity auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("send_card_entity unexpected error: chat_id=%s", chat_id)
            return {"code": -1, "msg": str(exc), "data": None}

    def update_card_stream(
        self,
        card_id: str,
        element_id: str,
        content: str,
        sequence: int,
    ) -> dict[str, Any]:
        """Stream-update card element content via cardkit API.

        Parameters
        ----------
        card_id:
            The card entity ID (1-20 chars), NOT message_id.
        element_id:
            The element within the card to update.
        content:
            The new full text content.
        sequence:
            Monotonic sequence number for the stream.
        """
        logger.debug(
            "update_card_stream start: card_id=%s element_id=%s sequence=%s content_len=%s",
            card_id,
            element_id,
            sequence,
            len(content),
        )
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
            response = requests.put(url, headers=headers, json=payload, timeout=30)
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
            return result
        except requests.HTTPError as exc:
            logger.error(
                "update_card_stream HTTP error: %s card_id=%s element_id=%s",
                exc,
                card_id,
                element_id,
            )
            return {"code": exc.response.status_code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception(
                "update_card_stream unexpected error: card_id=%s element_id=%s",
                card_id,
                element_id,
            )
            return {"code": -1, "msg": str(exc), "data": None}

    def patch_message(
        self,
        message_id: str,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        """Update an existing message (including interactive cards).

        Uses the Feishu ``im.message.patch`` OpenAPI to replace the entire
        message content. This is more reliable than cardkit stream updates
        because it does not require a separate card entity ID.

        Parameters
        ----------
        message_id:
            The ID of the message to update.
        content:
            The new message content dict. For interactive cards this should be
            the card JSON payload.

        Returns
        -------
        dict
            API response with ``code``, ``msg``, and ``data``.
        """
        logger.debug("patch_message start: message_id=%s content_keys=%s", message_id, list(content.keys()))
        content_json = json.dumps(content, ensure_ascii=False)
        body = (
            PatchMessageRequestBodyBuilder()
            .content(content_json)
            .build()
        )
        req = (
            PatchMessageRequestBuilder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )

        try:
            resp = self._client.im.v1.message.patch(req)
            result = self._unwrap_response(resp)
            if not resp.success():
                logger.error(
                    "patch_message failed: code=%s msg=%s message_id=%s",
                    resp.code,
                    resp.msg,
                    message_id,
                )
            else:
                logger.info("patch_message success: message_id=%s", message_id)
            return result
        except ObtainAccessTokenException as exc:
            logger.error("patch_message auth error: %s", exc)
            return {"code": exc.code, "msg": str(exc), "data": None}
        except Exception as exc:
            logger.exception("patch_message unexpected error: message_id=%s", message_id)
            return {"code": -1, "msg": str(exc), "data": None}

    # --------------------------------------------------------------------- #
    # WebSocket
    # --------------------------------------------------------------------- #

    def start_ws(self, event_handler: lark.EventDispatcherHandler) -> None:
        """Start the WebSocket client to receive real-time events.

        Parameters
        ----------
        event_handler:
            A ``lark.EventDispatcherHandler`` built via ``build_event_handler``.
            This method blocks while the connection is alive.
        """
        ws_client = lark.ws.Client(
            app_id=self._app_id,
            app_secret=self._app_secret,
            event_handler=event_handler,
            domain=self._domain,
            auto_reconnect=True,
        )
        logger.info("Starting Lark WebSocket client for app_id=%s debug=%s", self._app_id, self._debug)
        ws_client.start()
