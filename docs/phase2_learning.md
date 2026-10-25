# Phase 2：反事实教师、双风险头与主动信息获取

Phase 2 不再把“当前答案正确”和“最终任务成功”视为同一个预测目标。

## Phase 2A

Verify Head 预测当前决策点的候选答案是否正确：

\[
p_{current}=P(\text{current answer correct}\mid s)
\]

Value Head 预测在给定后续策略下最终任务是否成功：

\[
p_{success}=P(\text{final task success}\mid s,\pi)
\]

训练记录必须分别保存 `current_answer` 和 `final_outcome` 标签。Verify Head
只能读取 state features；Value Head 读取 state features 和 policy/action features。
Oracle、gold answer 和 hidden-test 结果只能出现在标签或最终评测结果中。

SEND 教师搜索枚举或逆向删除 optional fragments，mandatory fragments 由协议层
自动加入。搜索目标按以下顺序比较：

1. 动作合法；
2. 最终任务成功；
3. 总 token 成本；
4. 通信 token；
5. 轮次；
6. optional fragment 数量。

若所有 SEND 子集均失败，结果必须为 `no_successful_send`，不能制造成功教师，
该状态转入 Phase 2B。

## Phase 2B

在相同快照上比较：

```text
STOP / SEND / REQUEST / RETRIEVE / REVISE
```

REQUEST、RETRIEVE 和 REVISE 必须通过 `InformationActionAdapter` 连接真实环境、
检索器或修订流程。没有适配器时动作非法，系统不会读取标准答案或未来图状态来
伪造检索结果。

Phase 2B 教师仍以最终任务 evaluator 作为成功标签，并把动作成本、模型 token、
通信 token 和额外轮次共同纳入比较。

## 代码入口

- `workflow_runtime/communication_action.py`：五类动作、SEND dry-run、closure 后预算。
- `workflow_runtime/replay_runner.py`：每个反事实动作使用隔离 Workflow 继续执行。
- `workflow_runtime/learning/schemas.py`：当前正确性与最终成功的分离标签。
- `workflow_runtime/learning/error_features.py`：部署特征与 oracle 泄漏检查。
- `workflow_runtime/learning/risk_model.py`：Verify/Value 双头网络。
- `workflow_runtime/learning/teacher_search.py`：Phase 2A 和 Phase 2B 教师搜索。
- `workflow_runtime/learning/artifacts.py`：JSONL 产物与 task-level split 审计。

## 外部环境契约

项目运行时不会假设所有任务都能检索或自动修订。领域适配器负责：

1. 从当前可观测状态构造合法动作；
2. 执行真实 REQUEST/RETRIEVE/REVISE；
3. 把新证据或修订结果写回 GraphStore；
4. 报告实际成本；
5. 保证未使用 gold answer、hidden tests 或未来状态。

每个反事实 rollout 都必须由新的 Workflow 实例开始，恢复同一个通信快照后再
应用动作，防止不同动作分支互相污染。
