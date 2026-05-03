import json
import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)
_NODE_CONFIG_PATH = os.getenv("NODE_LLM_CONFIG", "node_llm_config.json")
_config_cache: dict | None = None
_config_cache_ts: float = 0.0
_CONFIG_TTL = 30.0


def _load_node_config() -> dict:
    global _config_cache, _config_cache_ts
    now = time.monotonic()
    if _config_cache is not None and (now - _config_cache_ts) < _CONFIG_TTL:
        return _config_cache
    try:
        with open(_NODE_CONFIG_PATH) as f:
            _config_cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _config_cache = {}
    _config_cache_ts = now
    return _config_cache


def get_llm(**kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("VOLC_MODEL", "ep-20260422180225-zllc4"),
        api_key=os.getenv("VOLC_API_KEY"),
        base_url=os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        **kwargs,
    )


def get_llm_for_node(node_name: str, **kwargs) -> ChatOpenAI:
    node_cfg = _load_node_config().get(node_name) or {}
    return ChatOpenAI(
        model=node_cfg.get("model") or os.getenv("VOLC_MODEL", "ep-20260422180225-zllc4"),
        api_key=node_cfg.get("api_key") or os.getenv("VOLC_API_KEY"),
        base_url=node_cfg.get("base_url") or os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        **kwargs,
    )


def invoke_structured(node_name: str, schema: type[T], prompt: str, **kwargs) -> T:
    return invoke_structured_with_llm(get_llm_for_node(node_name, **kwargs), schema, prompt)


def invoke_structured_with_llm(llm: object, schema: type[T], prompt: str) -> T:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    response = llm.invoke(
        f"{prompt}\n\n只输出一个合法 JSON 对象，不要 Markdown，不要解释。\nJSON Schema：{schema_json}"
    )
    direct = _model_fields_from_object(response, schema)
    if direct:
        return schema.model_validate(direct)
    content = getattr(response, "content", response)
    return schema.model_validate(_parse_json_content(_content_text(content)))


def _model_fields_from_object(value: object, schema: type[BaseModel]) -> dict[str, object]:
    if isinstance(value, schema):
        return value.model_dump()
    if isinstance(value, BaseModel):
        return value.model_dump()
    data = getattr(value, "__dict__", None)
    if not isinstance(data, dict):
        return {}
    return {
        field: data[field]
        for field in schema.model_fields
        if field in data
    }


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content or "")


def _parse_json_content(text: str) -> object:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("LLM did not return valid JSON")
