# 产物更新路径设计

**日期**：2026-05-05  
**背景**：用户发送"修改这个 PPT（URL）风格换成绿色"时，系统走了新建路径，凭空生成了无关内容。根因是路由层没有"新建 vs 更新"的区分，且 URL 提取只覆盖 `/docx/` 和 `/wiki/`，漏掉了 `/slides/` 和 whiteboard。本设计修复三种产物（doc / whiteboard / slide）的更新路径。

---

## 一、路由层：`RouteDecision` 加 `update_targets`

### 变更

`RouteDecision` 新增字段：

```python
update_targets: list[str] = Field(
    default_factory=list,
    description="用户要修改的现有产物 URL 列表，新建时为空"
)
```

`ROUTER_PROMPT` 加规则：

> 当用户意图是修改/更新/调整/重新设计现有产物时，把消息里的产物 URL 填入 `update_targets`；新建时留空。

### 兜底验证

`_parse_route_payload` 对 `update_targets` 里的每个 URL 做正则验证：必须能在原始消息里找到，否则丢弃。防止 LLM 幻觉出不存在的 URL。

---

## 二、URL 提取：扩展到三种产物

### 变更

`_doc_refs_from_message` 改名为 `_artifact_refs_from_message`，正则扩展为：

```python
# doc / wiki（现有）
r"https?://[^\s<>()\"']+/(?:docx|wiki)/[A-Za-z0-9_-]+"
# slides
r"https?://[^\s<>()\"']+/slides/[A-Za-z0-9_-]+"
```

新增工具函数：

```python
def _token_from_artifact_url(url: str) -> tuple[str, str]:
    """返回 (kind, token)，kind 是 doc/whiteboard/slide，token 是产物标识符。"""
```

- `/docx/TOKEN` 或 `/wiki/TOKEN` → `("doc", TOKEN)`
- `/slides/TOKEN` → `("slide", TOKEN)`
- whiteboard token 嵌在 docx 文档里，不直接从 URL 提取，通过 `fetch_doc_content` 后解析

所有调用 `_doc_refs_from_message` 的地方改为 `_artifact_refs_from_message`。

---

## 三、读取现有产物内容

### doc

复用已有 `fetch_doc_content(url, uat)` → 返回 DocxXML 字符串。

### whiteboard

新增 `fetch_whiteboard_content(token: str, uat: str) -> str`：

```
lark-cli whiteboard +query --whiteboard-token TOKEN --output_as code --as user
```

返回 mermaid/DSL 字符串，失败返回空字符串。

### slide

新增 `fetch_slide_content(token: str, uat: str) -> str`：

```
lark-cli slides xml_presentations get --params '{"xml_presentation_id":"TOKEN"}' --as user
```

返回演示文稿 XML 字符串（包含所有页面内容和 slide_id），失败返回空字符串。

同时新增 `fetch_slide_ids(token: str, uat: str) -> list[str]`，从上述响应中解析出所有 slide_id，供删除时使用。

---

## 四、生成函数：更新模式

三个生成函数各加 `existing_content: str = ""` 参数。当非空时，prompt 切换到更新模式。

### `generate_doc_content`

更新模式 prompt 追加：

```
以下是现有文档内容（DocxXML），按用户要求修改，保留无需变更的部分：
{existing_content}
```

### `generate_whiteboard_mermaid`

更新模式 prompt 追加：

```
以下是现有白板内容（Mermaid），按用户要求修改，保留无需变更的部分：
{existing_content}
```

### `generate_slide_xml`

更新模式 prompt 追加：

```
以下是现有演示文稿的 XML 内容，按用户要求修改（如风格、颜色、内容调整），
保留无需变更的页面结构和文字：
{existing_content}
```

注意：更新模式下，`SLIDE_OUTLINE_PROMPT` 的内容规划逻辑仍然运行，但 LLM 应以现有内容为基础而非凭空生成。

---

## 五、写回现有产物

### doc 更新

新增 `update_doc_from_content(token: str, content: str, uat: str) -> SkillArtifact`：

```
lark-cli docs +update --doc TOKEN --mode overwrite --markdown CONTENT --as user
```

返回 `SkillArtifact`，`status` 为 `"updated"` 或 `"error"`，`url` 保持原始 URL。

### whiteboard 更新

新增 `update_whiteboard_from_mermaid(token: str, mermaid: str, uat: str) -> SkillArtifact`：

```
lark-cli whiteboard +update --whiteboard-token TOKEN --source - --input_format mermaid --overwrite --yes --as user
```

（逻辑与 `create_whiteboard_from_mermaid` 里的 `whiteboard +update` 调用相同，抽出独立函数。）

### slide 更新

新增 `update_slide_from_xml(token: str, slides_xml: str, uat: str) -> SkillArtifact`：

1. 调用 `fetch_slide_ids(token, uat)` 获取现有 slide_id 列表
2. 逐个调用 `xml_presentation.slide delete` 删除现有页面
3. 解析 `slides_xml`（JSON 数组）逐个调用 `xml_presentation.slide create` 插入新页面
4. 返回 `SkillArtifact`，`url` 保持原始 URL，`status` 为 `"updated"` 或 `"error"`

---

## 六、`_run_deterministic_artifacts` 分支

每个 step 的逻辑变为：

```python
update_url = _find_update_target(route.update_targets, kind=step)
if update_url and user_access_token:
    token = _token_from_artifact_url(update_url)[1]
    existing = fetch_XXX_content(token, uat)
    new_content = generate_XXX(message, existing_content=existing, context=context)
    artifact = update_XXX(token, new_content, uat)
else:
    # 现有创建路径不变
```

`_find_update_target(targets, kind)` 遍历 `update_targets`，找到 `_token_from_artifact_url(url)[0] == kind` 的第一个 URL。

更新成功后 artifact 的 `url` 保持原始 URL，`status` 为 `"updated"`。`SkillArtifact` 的 `status` 类型扩展为包含 `"updated"`。

---

## 七、不在本次范围内

- whiteboard token 嵌在 docx 里的情况（用户直接发 whiteboard token 而非 docx URL）：暂不处理，后续按需扩展
- `multi` 路由下同时更新多个产物：逻辑相同，自然支持
- slide 更新的乐观锁（`revision_id`）：使用默认 `-1`（最新版本），不做并发控制

---

## 八、受影响文件

| 文件 | 变更类型 |
|------|----------|
| `deep_agent/service.py` | 路由字段、URL 提取、分支逻辑 |
| `skills/lark_doc.py` | `generate_doc_content` 加 `existing_content`，新增 `update_doc_from_content` |
| `skills/lark_whiteboard.py` | `generate_whiteboard_mermaid` 加 `existing_content`，新增 `fetch_whiteboard_content`、`update_whiteboard_from_mermaid` |
| `skills/lark_slide.py` | `generate_slide_xml` 加 `existing_content`，新增 `fetch_slide_content`、`fetch_slide_ids`、`update_slide_from_xml` |
| `skills/base.py` | `SkillArtifact` status 类型扩展（如有类型注解） |
