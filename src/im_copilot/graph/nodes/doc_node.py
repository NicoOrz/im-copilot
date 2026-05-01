import logging
import os
from functools import lru_cache
from pathlib import Path

from im_copilot.lark_cli import run_lark_cli
from im_copilot.llm import get_llm_for_node
from im_copilot.state import PipelineState

logger = logging.getLogger(__name__)

_LARK_DOC_ROOT = Path(
    os.getenv("IM_COPILOT_LARK_DOC_SKILL_ROOT", "/home/claude_worker/.agents/skills/lark-doc")
)

DOC_PROMPT = """你是一位专业的飞书云文档撰写助手。请根据用户提供的原始内容和主题，生成一份完整的飞书 DocxXML 文档。

下面是 lark-doc skill 的完整 Markdown 资料。你必须遵循这些规则生成文档内容：

<lark_doc_skill>
{lark_doc_skill}
</lark_doc_skill>

用户原始请求：
{raw_message}

主题：{topic}

要求：
- 直接输出完整 DocxXML 内容，必须包含唯一 <title>
- 忠实反映原始材料的关键信息，不要添加原始材料中没有的内容
- 不要加代码块标记（```），不要添加额外解释
- 标签本身不要转义，只有标签内文本需要按 XML 规则转义
- 默认使用结构化 block 提升可读性，例如 callout、table、grid、checkbox、hr
- 不要添加额外解释"""


@lru_cache(maxsize=1)
def _load_lark_doc_skill() -> str:
    if not _LARK_DOC_ROOT.exists():
        logger.warning("lark-doc skill root not found: %s", _LARK_DOC_ROOT)
        return ""

    parts: list[str] = []
    paths = [
        path for path in _LARK_DOC_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() == ".md"
    ]
    for path in sorted(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("failed to read lark-doc skill file: path=%s error=%s", path, exc)
            continue
        rel_path = path.relative_to(_LARK_DOC_ROOT)
        parts.append(f"## {rel_path}\n{text}")

    content = "\n\n".join(parts)
    logger.info(
        "loaded lark-doc skill context: root=%s files=%s chars=%s",
        _LARK_DOC_ROOT,
        len(parts),
        len(content),
    )
    return content


def doc_node(state: PipelineState) -> dict:
    topic = state.get("intent_params", {}).get("topic", "未命名文档")
    raw_message = state.get("raw_message", "")
    uat = state.get("user_access_token", "")
    title = f"文档：{topic}"

    content = get_llm_for_node("doc").invoke(DOC_PROMPT.format(
        topic=topic,
        raw_message=raw_message,
        lark_doc_skill=_load_lark_doc_skill(),
    )).content.strip()

    result = {
        "kind": "doc",
        "title": title,
        "status": "draft",
        "preview": content,
        "token": "",
        "url": "",
    }

    try:
        resp = run_lark_cli([
            "docs", "+create",
            "--api-version", "v2",
            "--content", content,
            "--as", "user",
        ], uat=uat)
        doc_token = resp.get("data", {}).get("document", {}).get("document_id", "")
        if doc_token:
            url = f"https://www.feishu.cn/docx/{doc_token}"
            result.update({"status": "created", "token": doc_token, "url": url})
            logger.info("Created doc %s", doc_token)
        else:
            logger.error("doc +create returned no token: %s", resp)
    except Exception:
        logger.exception("doc API failed, falling back to draft")

    return {"artifacts": {**state.get("artifacts", {}), "doc": result}}
