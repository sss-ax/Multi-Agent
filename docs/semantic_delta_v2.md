# Semantic Delta v2

`semantic_delta_v2` is the runtime's only model-facing protocol. There is no compatibility path for the removed incremental Action protocol.

## Boundary

1. Dataset adapters create trusted Source nodes with explicit `origin=source` and `visibility_class=public`.
2. Gold answers, supporting-fact labels, and private tests live in `EvaluationContext` only.
3. Planner, Solver, and Critic each emit one role/domain-specific semantic action. They never emit graph IDs, node types, edges, completion flags, or routing hints.
4. The schema validator rejects extra fields and cross-role/domain operations. The kernel deterministically lowers valid semantic actions to its private Semantic Delta IR.
5. `SemanticDeltaValidator` checks role/type/relation permissions, visible source references, quote grounding, endpoint existence, cycles, receiver names, and fragment ownership.
6. The kernel sets completion and required outputs from `TaskSpec`; the model cannot claim stage completion.
7. `SemanticDeltaCompiler` builds a mutation on a complete GraphStore clone and swaps it into the canonical store only after every node and edge succeeds. It assigns logical IDs, versions, provenance, ownership, status, edge IDs, and routing while preserving branch ancestry.
8. `AgentGraphViewManager` grants only public inputs initially. Restricted visibility is deny-first for both initial and dynamic grants.
9. The kernel generates communication candidates and checks sender visibility, receiver legality, novelty, dependency closure, and receiver-relative marginal token cost before granting a fragment.
10. The Critic emits the task-specific `judge_numeric`, `audit_choice`, `diagnose_code`, or `judge_evidence` action. For multiple-choice tasks, `audit_choice` only checks the Solver's existing claims; it cannot emit an independent answer or replacement option. The kernel derives verification/invalidation state from the audit coverage and statuses, then chooses repair/finalization routing.
11. Finalization is deterministic: no Final Solver model call occurs. The kernel copies only the result/code targeted by a verified verification edge.
12. Private code tests return only typed failure control data. Test expressions, inputs, expected/actual values, tracebacks, stdout, and stderr are not copied into model-visible runtime nodes.

## Enabling it

```bash
python main.py \
  --model-path /path/to/model \
  --task-type numeric_solve \
  --communication-policy send_all
```

The three comparable communication baselines are `send_all`, `mandatory_only`, and `model_selected`. In v2, `model_selected` operates on kernel-derived eligible fragments; unknown, non-optional, invisible, duplicate, or over-budget fragments are rejected by the kernel.

`semantic_delta_v2` is fail-closed and currently supports the four registered task families used by the five target datasets:

| Dataset | TaskSpec |
|---|---|
| GSM8K | `numeric_solve` |
| MMLU-Pro | `multiple_choice` |
| MBPP | `code_generation` |
| HumanEval | `code_generation` |
| HotpotQA | `multihop_qa` |

Other task types are rejected until they receive an explicit TaskSpec and end-to-end tests.

## Stateless input compression

The local Transformers backend does not retain a server-side conversation or
KV cache.  Protocol retries therefore use a standalone Action-repair prompt:
the rejected output, validator diagnostic, contract, and only explicitly
referenced graph nodes.  They do not replay the full role context.  Telemetry
reports `retry_input_tokens`, `retry_full_replay_equivalent_tokens`, and
`retry_saved_input_tokens` so the physical prefill saving is auditable.

In `role_aware` mode, multihop source rendering is also question-guided and
bounded without consulting gold supporting-fact labels.  The default is 48
`sentence_source` nodes per call; already-grounded sources are retained.  The
domain evaluator accepts `--multihop-sentence-limit N` to change the bound or
`0` to disable it.  The
other registered task families keep their complete, comparatively small
required Source sets.

## Model payload shapes

Planner:

```json
{
  "op": "plan",
  "facts": [{"content": "Alice has 2 apples", "source": "task", "quote": "Alice has 2 apples"}],
  "strategy": "add the stated quantities"
}
```

Solver uses exactly one domain action:

- numeric: `solve_numeric(expression, value)`，其中 `expression` 和 `value` 都是非空字符串
- multiple choice: `choose_option(answer, slot_analysis, eliminated_options, confidence)`；`slot_analysis` 必须逐槽给出所选词和理由，`eliminated_options` 必须至少解释一个被排除选项，`confidence` 为 `0/0.25/0.5/0.75/0.9/1`
- code: `submit_code(code)`
- multihop QA: `answer_with_evidence(answer, bridge_entity, citations, reasoning)`，其中 `citations` 至少包含两个当前可见的 `sentence_source` 引用

Planner 和 Solver 使用完整 token-level Schema 约束解码。运行时固定 JSON 标点、字段名、字段顺序、类型、枚举和终止位置；选择题标签与证据引用从当前 Agent 可见图动态生成。模型只生成字符串槽中的语义内容。

Critic:

```json
{"op":"judge_numeric","model_verdict":"consistent","error_type":"none","error_location":"","reason":"the calculation is consistent","repair_instruction":""}
```

The payload must not contain node IDs, temporary IDs, versions, owners, graph status, visibility, node types, edges, completion flags, routing hints, gold labels, or private test bodies. Source citations are stable Source Graph logical references resolved by the kernel.

Removed operations such as `add_fact`, `set_result`, model-facing `verify`, `answer`, and `propose_delta` are rejected. Final Solver has no model-facing action.
