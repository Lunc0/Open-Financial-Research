# 运行时引导闭环、显式研究计划与结构化可追踪性

本文说明本项目相对上游 Open Deep Research 新增的三项核心机制：运行时引导闭环、显式研究计划和结构化可追踪性。三者不是彼此独立的功能，而是一条完整控制链：用户在研究运行期间提出新要求，系统将要求结构化地写入状态，由 Supervisor 转换为可执行计划，再把研究行为和状态变化记录为机器可读事件。

## 1. 解决的问题

上游流程适合“提交问题，然后等待报告”的一次性研究。如果研究已经开始，用户又补充“只看近三年”“优先使用 SEC 原始文件”或“不要再研究估值，改查现金流”，仅把新文本追加到消息历史会产生三个问题：

1. 模型可能读到了要求，却没有真正修改任务分解。
2. 无法判断一条新要求是否已经被处理。
3. UI 和调用方只能看到零散消息，无法可靠展示当前计划和执行进度。

本项目因此引入三层状态：

```text
用户新增要求
    │
    ▼
user_guidance              记录“用户要求了什么、是否已处理”
    │
    ▼ UpdateResearchPlan
research_plan              记录“系统准备执行什么、当前执行到哪里”
    │
    ├── ConductResearch     计划项与真实研究任务绑定
    │
    ▼
trace_events / node_status 记录“发生过什么、节点现在是什么状态”
```

设计边界是：LLM 负责理解要求和提出计划变更，代码负责校验、合并状态、阻止非法执行和记录事实。

## 2. 核心状态对象

### 2.1 `UserGuidanceItem`

`state.user_guidance` 保存运行中收到的结构化用户引导：

```python
class UserGuidanceItem(TypedDict):
    id: str
    content: str
    created_at: str
    source: str
    priority: int
    applies_to: Optional[str]
    status: str
```

字段语义：

- `id`：稳定标识，用于把引导与计划变更关联。
- `content`：用户新增、修改、收窄或纠偏的要求。
- `created_at`：进入系统的时间。
- `source`：来源，例如服务层用户输入或 `studio_chat`。
- `priority`：数值越大越优先，Supervisor 读取前会排序。
- `applies_to`：希望影响的阶段，默认面向 Supervisor。
- `status`：通常从 `pending` 进入 `applied`；无法处理时进入 `error`。

它回答的是：“用户后来又要求了什么，这条要求处理完了吗？”

### 2.2 `ResearchPlanItem`

`state.research_plan` 保存显式研究计划：

```python
class ResearchPlanItem(TypedDict):
    id: str
    title: str
    description: str
    status: str
    priority: int
    created_from_guidance_id: Optional[str]
```

计划项状态包括：

```text
pending → running → completed
    ├──────────────→ cancelled
    ├──────────────→ superseded
    └──────────────→ error
```

只有 `pending` 和 `running` 是可执行状态。`created_from_guidance_id` 提供最直接的来源关联；批量吸收多个引导时，`ResearchPlanChange.guidance_ids` 用于声明本次变更处理了哪些引导。

它回答的是：“系统准备研究什么，每项工作处于什么阶段？”

### 2.3 `TraceEvent` 与 `NodeStatus`

可追踪性由两种语义不同的数据组成。

`TraceEvent` 是只追加的事件日志：

```python
class TraceEvent(TypedDict):
    id: str
    timestamp: str
    node: str
    event_type: str
    summary: str
    details: dict[str, Any]
```

`NodeStatus` 是某个节点的最新状态快照：

```python
class NodeStatus(TypedDict):
    status: str
    summary: str
    updated_at: str
    details: dict[str, Any]
```

两者的区别类似“流水账”和“仪表盘”：

- `trace_events` 使用列表追加 reducer，保留历史顺序，适合审计、回放和诊断。
- `node_status` 使用字典合并 reducer，同名节点的新状态覆盖旧状态，适合 UI 展示当前进度。

## 3. 运行时引导如何进入图

系统提供两条入口。

### 3.1 服务层入口

聊天后端可以调用 `append_runtime_guidance` 或异步版本 `aappend_runtime_guidance`。辅助函数会：

1. 校验 `thread_id`，确保更新的是已有研究线程。
2. 把一条或多条文本规范化为 `UserGuidanceItem`。
3. 通过图的 state update API 追加到 `state.user_guidance`。
4. 同时追加 `guidance_received` trace event。

这里使用“追加”而不是覆盖，避免并发到达的新要求互相吞掉。

### 3.2 LangGraph Studio Chat 入口

已有 `research_brief` 的线程再次收到 HumanMessage 时，`clarify_with_user` 不会重写原始 brief。`collect_unhandled_chat_guidance` 会把尚未处理的聊天消息转换为：

```text
id = guidance_from_<message_id>
source = studio_chat
status = pending
```

消息 ID 会加入 `handled_guidance_message_ids`，防止同一条 Studio 消息在图再次运行时被重复转换。随后流程直接回到 `research_supervisor`。

## 4. `user_guidance` 的状态合并语义

`user_guidance_reducer` 不依赖普通列表拼接，而是支持显式操作：

- `append`：追加新引导。
- `override`：整体替换，主要用于恢复或测试。
- `mark_status`：按 ID 批量修改状态。

普通输入默认按追加处理。这样既能支持外部服务动态写入，又能让 Supervisor 在计划变更成功后只更新对应引导的状态。

关键点是：收到引导不等于应用引导。只有计划变更真正成功，代码才把关联的 guidance ID 标记为 `applied`。

## 5. Supervisor 如何消费运行时引导

每轮 Supervisor 执行前，系统会：

1. 规范化 `user_guidance`。
2. 按 `priority` 从高到低排序。
3. 格式化引导内容及其 ID、来源和状态。
4. 同时格式化当前 `research_plan`。
5. 将两部分加入 Supervisor 上下文。

因此，模型同时看到“用户的新要求”和“当前已有计划”，能够选择增加、替换、取消或调整计划项，而不是从头生成一份与当前执行状态脱节的新计划。

## 6. 显式研究计划协议

Supervisor 通过 `UpdateResearchPlan` 提交结构化变更。单条 `ResearchPlanChange` 支持：

- `add`：新增计划项。
- `replace`：把旧计划项标记为 `superseded`，并建立替代项。
- `update_status`：更新一个或多个计划项状态。
- `update_priority`：调整一个或多个计划项优先级。

`apply_research_plan_changes_with_metadata` 在代码层应用这些变更，并返回：

```text
(更新后的 research_plan, 真正被成功吸收的 guidance_ids)
```

只有变更确实作用于计划项时，相关 guidance ID 才会出现在第二个返回值中。随后 `supervisor_tools`：

1. 用 `override` 一次性写回完整计划，避免重复合并。
2. 把成功吸收的引导标记为 `applied`。
3. 清零 `pending_guidance_completion_attempts`。
4. 写入 `plan_updated` trace event，保存 rationale、changes 和更新后的计划快照。

## 7. 计划与真实研究任务的强绑定

`ConductResearch` 新增必填参数 `plan_item_id`：

```python
class ConductResearch(BaseModel):
    research_topic: str
    plan_item_id: str
```

执行前，`validate_conduct_research_plan_item` 会检查：

1. 是否提供 `plan_item_id`。
2. ID 是否存在于 `state.research_plan`。
3. 对应项是否为 `pending` 或 `running`。

非法调用不会启动研究子图，而是返回 ToolMessage 并记录 `research_task_skipped`。合法调用会先把计划项更新为 `running`，再并发启动研究任务。任务结束后，代码根据执行结果更新计划状态，而不是让 LLM 自行声称“已经完成”。

这一约束把模型生成的任务分派与可观察状态连接起来：每个真实研究任务都必须能追溯到一条计划项。

## 8. 阻止“看到了引导但直接结束”

闭环最关键的保护发生在 Supervisor 准备结束时。以下任一条件原本都可能结束研究阶段：

- 达到最大迭代次数。
- 模型没有再调用工具。
- 模型调用 `ResearchComplete`。

代码会先检查是否仍有 `pending` guidance。如果有，则不立即结束，而是：

1. 生成反馈消息，明确要求先调用 `UpdateResearchPlan`。
2. 回到 `supervisor`。
3. 增加 `pending_guidance_completion_attempts`。
4. 记录 `guidance_pending_before_completion`。
5. 将 Supervisor 节点状态保持为 `running`。

这相当于一个代码级完成门：模型不能只在自然语言里表示“知道了”，必须通过结构化计划操作证明已经处理。

为了避免模型持续失败造成死循环，重试达到上限后系统会：

- 将仍未处理的 guidance 标记为 `error`。
- 记录 `guidance_application_failed`。
- 清零计数器并允许流程结束。

因此该机制同时满足“尽量不漏掉用户要求”和“失败时可终止”。

## 9. 结构化可追踪性如何覆盖整个流程

`build_trace_event` 和 `build_node_status` 是统一构造器，负责生成时间、节点和运行上下文一致的数据。当前主要事件包括：

| 阶段 | 典型事件 | 表达的事实 |
| --- | --- | --- |
| 输入 | `guidance_received` | 收到并结构化了新引导 |
| 澄清 | `clarification_requested` | 研究范围不足，需要用户回答 |
| Supervisor | `guidance_pending` | 本轮规划读到了待处理引导 |
| 计划 | `plan_updated` | 显式计划变更已由代码应用 |
| 完成门 | `guidance_pending_before_completion` | 因未处理引导阻止提前结束 |
| 失败兜底 | `guidance_application_failed` | 多次规划失败，引导进入错误状态 |
| 分派 | `research_task_skipped` | 计划 ID、状态或并发限制不合法 |
| 研究 | 工具调用与研究完成事件 | 子研究任务实际执行情况 |
| 压缩 | 来源与压缩事件 | 研究笔记压缩及轻量来源信息 |
| 报告 | 报告生成事件 | 最终报告的开始、完成或错误 |

`details` 可以保存关联 ID、失败原因、计划快照、迭代次数等结构化信息。它不是完整的分布式 tracing 系统，也不应存放密钥或大段原始网页；它的目标是让调用方不解析自然语言消息也能回答：

- 哪条用户引导还没处理？
- 哪次计划更新吸收了它？
- 哪个研究任务对应哪个计划项？
- 为什么某个任务没有执行？
- 当前哪个节点仍在运行或发生错误？

## 10. 一次完整闭环示例

假设研究已开始，用户新增要求：“不要再比较同行估值，改为重点核查最近三年的经营现金流和资本开支。”

### 步骤 1：写入引导

```json
{
  "id": "guidance_42",
  "content": "不要再比较同行估值，改为重点核查最近三年的经营现金流和资本开支。",
  "source": "user",
  "priority": 10,
  "applies_to": "supervisor",
  "status": "pending"
}
```

同时产生 `guidance_received`。

### 步骤 2：Supervisor 提交计划变更

```json
{
  "changes": [
    {
      "operation": "replace",
      "supersedes_ids": ["plan_peer_valuation"],
      "plan_item_id": "plan_cashflow_capex",
      "title": "核查三年现金流与资本开支",
      "description": "以公司原始披露为主，比较最近三年的经营现金流、资本开支及自由现金流变化。",
      "status": "pending",
      "priority": 10,
      "created_from_guidance_id": "guidance_42",
      "guidance_ids": ["guidance_42"]
    }
  ],
  "rationale": "用户明确取消同行估值方向，并要求改查现金流和资本开支。"
}
```

代码将旧项改为 `superseded`，加入新项，将 `guidance_42` 标记为 `applied`，并记录 `plan_updated`。

### 步骤 3：分派真实研究任务

```json
{
  "research_topic": "核查公司最近三年的经营现金流、资本开支和自由现金流变化，并说明口径。",
  "plan_item_id": "plan_cashflow_capex"
}
```

校验通过后，`plan_cashflow_capex` 进入 `running`，研究子图开始执行。若 ID 不存在，则任务不会运行，并产生 `research_task_skipped`。

### 步骤 4：完成与展示

任务完成后，计划项进入 `completed`。UI 可分别使用：

- `research_plan` 渲染计划清单和进度。
- `node_status` 渲染当前阶段。
- `trace_events` 渲染时间线和诊断详情。
- `user_guidance` 展示用户补充要求是否已应用。

## 11. 关键代码入口

| 机制 | 代码入口 |
| --- | --- |
| 引导数据结构与 reducer | `src/open_deep_research/state.py` |
| 服务层追加引导 | `src/open_deep_research/runtime_guidance.py` |
| Trace/NodeStatus 构造器 | `src/open_deep_research/observability.py` |
| Studio 消息转引导 | `collect_unhandled_chat_guidance` |
| Supervisor 上下文注入 | `get_user_guidance_context`、`get_research_plan_context` |
| 完成门与失败兜底 | `supervisor_tools` |
| 计划变更执行 | `apply_research_plan_changes_with_metadata` |
| 研究任务计划校验 | `validate_conduct_research_plan_item` |
| 主图与子图编排 | `src/open_deep_research/deep_researcher.py` |

## 12. 机制价值与限制

### 价值

- 用户可以在长时间研究过程中纠偏，而不必重启整个任务。
- 引导是否被处理具有显式状态，不依赖聊天文本猜测。
- 研究计划成为外部可读、可更新、可验证的数据结构。
- 真实任务必须引用计划项，减少“计划一套、执行另一套”。
- UI、日志和测试可以消费结构化状态，不需要解析 LLM 自然语言。
- 失败路径有事件和错误状态，便于定位模型规划失败还是工具执行失败。

### 当前限制

- `trace_events` 是应用状态内的轻量事件记录，不等同于完整的 OpenTelemetry 或跨服务链路追踪。
- 引导到计划的语义映射仍由 LLM 决定；代码保证协议和状态一致性，但不保证规划内容一定正确。
- 重试上限后的 `error` 是可终止兜底，调用方仍应把未处理引导展示给用户。
- 多客户端同时更新同一线程时，仍依赖 LangGraph checkpoint/state update 的一致性语义。
- `details` 应保持精简并避免存放密钥、Token、完整隐私数据或大段检索正文。

## 13. 一句话总结

这三项机制把“用户中途说了一句话”变成了可验证的 Agent 控制流程：引导先进入结构化状态，Supervisor 必须把它映射成显式计划，真实研究任务必须绑定计划项，所有关键变化再通过事件历史和节点快照对外可见。
