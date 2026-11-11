# HotpotQA AFlow 最小上下文 MVP

`graph_minctx` 保持 AFlow round 3 的 Operator 拓扑不变，只为
`AnswerGenerate` 从公开 HotpotQA context 构造 source artifact graph，并按
profile 选择无损的文档/句子子集。`ScEnsemble` 和 `Custom` 继续使用现有
candidate binding、dedup/reuse 和原生 Ensemble 行为。

## 运行示例

```bash
.venv/bin/python scripts/reproduce_aflow_paper.py \
  --dataset HotpotQA \
  --split validate \
  --workflow-round 3 \
  --model-path /root/models/models/qwen--Qwen2.5-7B-Instruct \
  --limit 50 \
  --runs 1 \
  --communication-mode graph_minctx \
  --communication-policy send_all \
  --aflow-content-dedup \
  --aflow-invocation-reuse \
  --aflow-context-policy dependency_minimal \
  --aflow-context-top-k-documents 4 \
  --aflow-context-fallback original \
  --output .runtime/aflow_minctx_hotpotqa/validate_dependency_minimal.json
```

可用策略为 `full`、`topk_documents` 和 `dependency_minimal`。`full` 用于
full-context parity；选择策略失败、超过预算或无法满足 question/evidence
契约时回退到原始 Operator kwargs。

每个 graph run 的 `aflow_effective_policy.reasoning_context` 和 graph telemetry
记录选择策略、节点数量、输入 token、节省 token、闭包节点和回退原因。
Gold `supporting_facts` 不会写入 GraphStore 或模型上下文。
