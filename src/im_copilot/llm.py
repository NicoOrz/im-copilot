import json
import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

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
