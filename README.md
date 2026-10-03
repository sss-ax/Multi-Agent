# Multi-Agent Workflow Runtime

当前运行时采用五层结构：

```text
LangGraph
  └── Planner / Solver / Tool / Critic / Repair / Finalizer
        ↓
GraphStore
  └── 节点版本、依赖边、分支快照、失效传播
        ↓
Agent Visibility View
  └── 每个 Agent 只看到 canonical graph 的一个子图
        ↓
Graph Delta Communication
  └── novelty policy + receiver-relative dependency closure
        ↓
ContextSlicer
  └── 按角色读取必要图状态，生成紧凑上下文
        ↓
Action Protocol
  └── 严格 JSON Action、角色权限、状态约束
        ↓
模型后端
  └── 当前提供无状态 Transformers callable；可替换为其他全提示后端
```

模型与 GraphStore 之间只使用增量 Action IR：

```text
模型每次只输出一个 Action JSON
        ↓
Action Validator
        ↓
Action Compiler
        ↓
GraphStore 节点和依赖边
```

例如：

```json
{"op":"add_fact","id":"A","value":2}
```

模型不能直接写入 `task`、节点版本或图边；这些由 Runtime 根据角色权限和 Action 内容确定性生成。每个角色连续产生 Action，直到输出：

```json
{"op":"done"}
```

当前 Transformers 后端是无状态调用：每次请求都输入完整的当前提示，不保留跨调用因果状态，也不做前缀复用。实验目标是通过 GraphStore、Agent 可见性视图、图增量通信和结构化 Action 减少发送给模型的语义上下文 token，同时尽量保持推理质量。

Optimized workflow 使用单一 canonical `GraphStore`，不会为每个 Agent 复制整张图。每个 Agent 维护一个 `AgentGraphView`，通信本质是给 receiver 授权看见一批 canonical node/edge。Policy 只选择候选 root nodes，最终发送集合总是经过 receiver-relative dependency closure：

```text
selected_roots = policy(DeltaCandidate | sender_view, receiver_view)
Delta_send = DependencyClosure(selected_roots, receiver_view)
receiver_view += Delta_send
```

如果 receiver 已经拥有依赖节点，closure 不会重复发送；如果 receiver 缺少依赖，closure 会自动补齐，避免发送断裂图。

运行时支持工具和分支：Solver 可以通过 `call_tool` 调用 `ToolRegistry` 中的白名单工具，工具请求和结果都会写入 GraphStore；`GraphStore.create_branch`/`merge_branch` 提供隔离的分支版本空间，`LangGraphWorkflow.fork` 会生成独立的 branch workflow。

## 运行时边界

- LangGraph 负责工作流节点、条件路由、重试和 checkpoint。
- `GraphStore` 负责语义状态、版本和依赖边。
- `AgentGraphView` 负责每个 Agent 的可见节点/边集合；跨 Agent 通信是 visibility grant，不复制图节点。
- `Graph Delta Communication` 负责从新增节点抽取 exportable candidates、做 novelty 判断、选择 root nodes，并自动补齐 receiver 缺失依赖。
- `ContextSlicer` 负责按角色读取必要图状态，避免把完整历史反复塞回模型。
- `Action IR` 负责模型到 Runtime 的最小增量操作。
- `ActionCompiler` 负责将合法 Action 编译成 GraphStore 更新。
- `TransformersModel` 是无状态全提示后端，只统计真实输入 token、输出 token 和 forward 次数。
- `workflow_runtime/tools.py`：白名单工具注册、参数校验和安全 calculator。
- `workflow_runtime/native_agents.py`：native Agent 身份、私有记忆、工具权限和生命周期。
- `workflow_runtime/message_bus.py`：native direct/broadcast 消息、mailbox、ack 和 transcript。
- `workflow_runtime/native_tooling.py`：native 工具请求解析、领域默认工具选择和结果模型。
- `GraphStore.create_branch/merge_branch`：分支 fork、隔离写入和带 provenance 的合并。
- 模型角色不再使用 LoRA adapter。

## Action 约束解码

每次 Action 生成前，运行时根据 GraphStore 当前边界建立状态约束。例如 numeric_solve 按 `declare_query -> add_fact -> add_plan_step -> done` 推进。Transformers 后端在逐 token decode 时对 `op` 分支进行 logits mask；对于题目中可确定的 numeric fact 和 plan，还会直接约束为合法的完整 Action 序列。ActionCompiler 仍保留字段、值和图状态校验，约束解码不是语义正确性的替代品。

Graph Memory embedding、Teacher/Student 训练和旧的角色 LoRA 训练脚本不属于当前推理入口；历史数据和模型产物暂不自动删除。

## 安装

```bash
python3 -m pip install -r requirements-train.txt
```

## 运行 LangGraph 工作流

```bash
python3 main.py \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --task-type numeric_solve \
  --task "A has 2 items and B has 3 items. What is the total?" \
  --communication-policy closure_aware_heuristic
```

带参考答案运行评估：

```bash
python3 main.py \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --task-type numeric_solve \
  --task "A has 2 items and B has 3 items. What is the total?" \
  --reference-answer 5 \
  --answer-tolerance 1e-6 \
  --log-path .runtime/logs/numeric-001.jsonl
```

`--reference-answer` 接受 JSON 标量、字符串或对象。数值任务使用绝对/相对容差；其他任务使用规范化文本或规范化 JSON 比较。未提供参考答案时，结果为 `not_evaluated`，不会计入错误率。日志中的 `answer_evaluation` 记录单条结果，`workflow_summary` 汇总 `evaluated_answers`、`correct_answers` 和 `incorrect_answers`。

可用任务类型：

```text
numeric_solve
numeric_comparison
table_qa
multihop_qa
multiple_choice
code_generation
marble_research
marble_bargaining
marble_database
```

## 代码入口

- `workflow_runtime/langgraph_workflow.py`：LangGraph 工作流和节点。
- `workflow_runtime/protocol.py`：严格的增量 Action 协议和角色权限。
- `workflow_runtime/action_compiler.py`：Action 到 GraphStore 的确定性编译器。
- `workflow_runtime/agent_graph_view.py`：canonical graph 上的 Agent 可见性视图。
- `workflow_runtime/graph_delta.py`：图增量候选和闭包发送数据结构。
- `workflow_runtime/delta_extractor.py`：从新增 graph mutation 中抽取 exportable delta candidates。
- `workflow_runtime/delta_closure.py`：receiver-relative dependency closure。
- `workflow_runtime/delta_scoring.py`：结构化 novelty、redundancy、token 和图结构特征。
- `workflow_runtime/communication/`：`send_all`、`random_keep_75/50/25`、`closure_aware_heuristic` 通信策略。
- `workflow_runtime/model_backend.py`：无状态 Transformers 模型适配器和真实 token 统计。
- `workflow_runtime/graph_store.py`：版本化语义图。
- `benchmarks/multiagentbench.py`：MultiAgentBench/MARBLE 任务规范化、领域映射和结果汇总。
- `benchmarks/marble_adapter.py`：将 native message trajectory、环境快照和 Agent 视图适配到官方 MARBLE Evaluator。
- `scripts/run_multiagentbench.py`：对同一任务批量运行 `native_langgraph` 与 `optimized`，保存 JSONL 结果、轨迹和报告。
- `scripts/evaluate_domain_workflow.py`：领域数据评估入口，支持 GSM8K、TAT-QA、HotpotQA、MBPP、HumanEval 和 MMLU-Pro 这类可用标准答案或测试用例本地评分的数据集。

## MultiAgentBench / MARBLE 对照实验

当前运行时把官方任务领域分别映射为：

```text
coding     -> code_generation
research   -> marble_research
bargaining -> marble_bargaining
database   -> marble_database
```

后三类使用独立的 GraphStore 任务边界和统一的
`result -> verification -> final_answer` 产物，不会被伪装成 coding 任务。每个任务都会保存 `trajectory.json`，其中包含可供官方 MARBLE Evaluator 消费的适配轨迹。未启用官方评测时，报告中的 `milestone_proxy_score` 只是本运行时的阶段完成代理值，不等于官方 MARBLE KPI。

使用官方 coding JSONL 或自定义 JSON/JSONL/YAML manifest：

```bash
.venv/bin/python scripts/run_multiagentbench.py \
  --input /path/to/MARBLE/marble/environments/coding_utils/assets/benchmark.jsonl \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --limit 2 \
  --output-dir .runtime/multiagentbench/coding \
  --overwrite
```

如果目标是官方 MARBLE 的通信、规划协作和 KPI 评分，显式启用官方
Evaluator（需要安装 `requirements-train.txt` 中的 MARBLE 评测依赖，并配置
judge LLM 的凭据）：

```bash
.venv/bin/python scripts/run_multiagentbench.py \
  --input /path/to/MARBLE/multiagentbench/research/research_main.jsonl \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --execution-mode native_langgraph \
  --official-marble \
  --marble-evaluator-model gpt-4o \
  --output-dir .runtime/multiagentbench/research \
  --overwrite
```

`--official-marble` 会调用上游 `marble.evaluator.Evaluator` 的
`evaluate_communication`、`evaluate_planning` 和 `evaluate_kpi`，并按领域调用
research/world/database/code evaluator。官方输入会写入每个任务的
`trajectory.json` 中的 `marble_trajectory`，官方结果写入
`official_marble`；如果评测依赖或 judge LLM 不可用，结果会保留
`official_marble_error`，使用 `--strict-official-marble` 可将其变成失败。
官方评分只对 `native_langgraph` 开启，因为 optimized workflow 不产生同一套
message-bus trajectory。

默认每个任务运行两个模式：

```text
native_langgraph  自然语言多代理 LangGraph 基线，不使用 GraphStore/Action 协议
optimized         GraphStore 可见性视图、图增量通信、严格 Action 协议和状态约束
```

Optimized 模式可选择图增量通信策略：

```text
send_all                  发送所有 exportable candidate roots，用作上界
random_keep_75            随机保留 75% OPTIONAL roots，然后做 receiver-relative closure
random_keep_50            随机保留 50% OPTIONAL roots，然后做 receiver-relative closure
random_keep_25            随机保留 25% OPTIONAL roots，然后做 receiver-relative closure
closure_aware_heuristic   基于 novelty、结构特征和 token cost 选择 roots，然后做 receiver-relative closure
```

命令行示例：

```bash
.venv/bin/python scripts/evaluate_domain_workflow.py \
  --domain gsm8k \
  --data-path data/gsm8k/test.jsonl \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --execution-mode optimized \
  --communication-policy closure_aware_heuristic \
  --communication-seed 0 \
  --communication-budget-tokens 256 \
  --limit 100 \
  --output .runtime/evals/gsm8k_heuristic_100.json
```

`native_langgraph` 中每个角色由独立的 `NativeAgent` 实例承载，而不是
直接把角色名传给同一个 workflow callable。每个 Agent 都有独立的
`agent_id`、短期记忆、工具权限、配置和生命周期状态。默认情况下多个
Agent 可以共享一个已加载的模型 provider 以避免重复占用 GPU；如果需要
完全不同的后端，可以通过 `NativeLangGraphWorkflow(agent_models=...)`
按角色注入模型 callable，或者直接传入 `agents`。

Native Solver 需要确定性工具时可以发送独立的工具信封（它不是 Action IR）：

```text
<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>
```

Tool 节点会通过白名单 `ToolRegistry` 执行请求，并把结构化结果、失败信息、
重试次数和 `environment_state` 写入 native state；工具失败会通过消息总线
回传 Solver，进入 repair → tool 重试链。默认领域工具包括 `calculator`、
`python_syntax_check` 和 `json_validate`，不会执行无约束的任意 Python 代码。

Critic 的修复路由只接受结构化 verdict，不解析自然语言关键词：

```text
<critic_verdict>{"verdict":"needs_repair","confidence":0.92,"target":"candidate","issues":["..."],"repair_instructions":"..."}</critic_verdict>
```

`needs_repair`/`reject` 只有在置信度达到配置阈值时才进入 repair；缺失或非法
verdict 会被标记为协议无效并跳过自动修复。

结果位于：

```text
.runtime/multiagentbench/coding/results.jsonl
.runtime/multiagentbench/coding/report.json
.runtime/multiagentbench/coding/<mode>/<task_id>/trajectory.json
```

报告同时记录运行完成率、参考答案准确率（若任务提供参考答案）、逻辑输入 token、物理输入 token、输出 token、图增量候选/发送 token、延迟和重试次数。没有参考答案时，`answer_accuracy` 保持为空，不能把运行完成率当作正确率。
