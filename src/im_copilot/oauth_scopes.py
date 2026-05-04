from __future__ import annotations


USER_OAUTH_SCOPES = (
    "offline_access",
    "contact:user.base:readonly",
    "calendar:calendar.event:create",
    "calendar:calendar.event:update",
    "docs:document.content:read",
    "docx:document:create",
    "docx:document:readonly",
    "docx:document:write_only",
    "docx:document",
    "drive:drive",
    "board:whiteboard:node:create",
    "board:whiteboard:node:delete",
    "slides:presentation:read",
    "slides:presentation:create",
    "slides:presentation:write_only",
    "slides:presentation:update",
    "wiki:wiki",
)


def user_oauth_scope_string() -> str:
    return " ".join(USER_OAUTH_SCOPES)
