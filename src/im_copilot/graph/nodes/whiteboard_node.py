import logging
import re

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

WHITEBOARD_PROMPT = """你是一位专业的可视化设计助手。请根据用户提供的原始内容和主题，生成 Mermaid 格式的图表。

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 直接输出 Mermaid 代码，不要加代码块标记（```）
- 根据内容选择合适的图类型（flowchart、sequenceDiagram、classDiagram、mindmap 等）
- 忠实反映原始材料的关键信息和关系
- 节点文字使用中文，简洁清晰
- 不要添加额外解释"""


def _get_llm():
    if not hasattr(_get_llm, "_instance"):
        _get_llm._instance = get_llm()
    return _get_llm._instance


def whiteboard_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名白板")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"白板：{topic}"

    mermaid = _get_llm().invoke(WHITEBOARD_PROMPT.format(topic=topic, raw_message=raw_message)).content.strip()

    result = {
        "kind": "whiteboard",
        "title": title,
        "status": "draft",
        "preview": mermaid,
        "token": "",
        "url": "",
    }

    try:
        # Step 1: create a docx wiki node
        resp = run_lark_cli([
            "wiki", "+node-create",
            "--title", title,
            "--obj-type", "docx",
            "--as", "user",
        ], uat=uat)
        obj_token = resp.get("data", {}).get("obj_token", "")
        if not obj_token:
            logger.error("wiki +node-create returned no obj_token: %s", resp)
            return {"artifacts": {**state.get("artifacts", {}), "whiteboard": result}}

        # Step 2: insert blank whiteboard block
        run_lark_cli([
            "docs", "+update",
            "--api-version", "v2",
            "--doc", obj_token,
            "--markdown", "",
            "--mode", "overwrite",
            "--as", "user",
        ], uat=uat)

        # Step 3: fetch doc to extract whiteboard token
        fetch_resp = run_lark_cli([
            "docs", "+fetch",
            "--api-version", "v2",
            "--doc", obj_token,
            "--as", "user",
        ], uat=uat)
        content = fetch_resp.get("data", {}).get("document", {}).get("content", "")
        match = re.search(r'token="([^"]+)"', content)
        wb_token = match.group(1) if match else ""

        if wb_token:
            # Step 4: write Mermaid content
            run_lark_cli([
                "whiteboard", "+update",
                "--whiteboard-token", wb_token,
                "--source", "-",
                "--input_format", "mermaid",
                "--overwrite",
                "--as", "user",
            ], uat=uat, stdin=mermaid)
            url = f"https://www.feishu.cn/docx/{obj_token}"
            result.update({"status": "created", "token": wb_token, "url": url})
            logger.info("Created whiteboard %s", wb_token)
        else:
            # No whiteboard block found, just return the doc link
            url = f"https://www.feishu.cn/docx/{obj_token}"
            result.update({"status": "created", "token": obj_token, "url": url})
    except Exception:
        logger.exception("whiteboard API failed, falling back to draft")

    return {"artifacts": {**state.get("artifacts", {}), "whiteboard": result}}
