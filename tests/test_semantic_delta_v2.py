import pytest

from workflow_runtime import (
    AgentGraphViewManager,
    CodeVerifier,
    EvaluationContext,
    GraphStore,
    NodeOrigin,
    SemanticDeltaCompiler,
    SemanticDeltaValidationError,
    VisibilityClass,
    node_visible_to_role,
)
from workflow_runtime.communication import make_communication_policy
from workflow_runtime.communication_replay import graph_state_to_dict
from workflow_runtime.action_compiler import ActionCompiler
from workflow_runtime.choice_audit import build_choice_audit_manifest
from workflow_runtime.langgraph_workflow import LangGraphWorkflow
from workflow_runtime.protocol import validate_action
from workflow_runtime.semantic_fragments import fragments_for_node


def _store():
    store = GraphStore()
    task = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="task",
        node_type="task_source",
        content="Alice has 2 apples and Bob has 3 apples.",
        owner="dataset",
        created_by_role="dataset",
        provenance={
            "origin": NodeOrigin.SOURCE,
            "trust": "trusted_input",
            "visibility_class": VisibilityClass.PUBLIC,
        },
    )
    return store, task


def test_semantic_delta_commits_grounded_nodes_and_kernel_owned_metadata():
    store, task = _store()
    compiler = SemanticDeltaCompiler(
        store,
        task_id="delta-task",
        task_type="numeric_solve",
    )
    result = compiler.apply(
        "planner",
        {
            "nodes": [
                {
                    "temp_id": "f1",
                    "type": "fact",
                    "content": {"subject": "Alice", "quantity": 2},
                    "grounding": [{"source": task.node_id, "quote": "Alice has 2 apples"}],
                },
                {
                    "temp_id": "p1",
                    "type": "plan_step",
                    "content": {"operation": "add", "inputs": ["f1"]},
                    "grounding": [],
                },
            ],
            "edges": [{"source": "f1", "relation": "input_to", "target": "p1"}],
            "send_hints": [{"receiver": "solver", "fragments": ["f1"], "reason": "fact input"}],
            "complete": True,
        },
        visible_node_ids=[task.node_id],
    )

    fact = store.latest_valid("delta-task", "main", "fact_f1")
    plan = store.latest_valid("delta-task", "main", "plan_step_p1")
    boundary = store.latest_valid("delta-task", "main", "semantic_delta_planner_boundary")
    assert fact is not None and plan is not None and boundary is not None
    assert fact.provenance["origin"] == NodeOrigin.MODEL
    assert fact.provenance["visibility_class"] == VisibilityClass.ROLE_SCOPED
    assert fact.owner == "planner"
    assert result.temp_to_node_id["f1"] == fact.node_id
    assert any(edge.relation == "grounded_in" and edge.target == fact.node_id for edge in store.snapshot().edges)
    assert any(edge.relation == "input_to" and edge.target == plan.node_id for edge in store.snapshot().edges)


def test_invalid_grounding_is_atomic():
    store, task = _store()
    compiler = SemanticDeltaCompiler(store, task_id="delta-task", task_type="numeric_solve")
    before = graph_state_to_dict(store.snapshot())

    with pytest.raises(SemanticDeltaValidationError, match="quote does not occur"):
        compiler.apply(
            "planner",
            {
                "nodes": [{
                    "temp_id": "f1",
                    "type": "fact",
                    "content": {"quantity": 99},
                    "grounding": [{"source": task.node_id, "quote": "not in source"}],
                }],
                "edges": [],
                "send_hints": [],
                "complete": True,
            },
            visible_node_ids=[task.node_id],
        )

    assert graph_state_to_dict(store.snapshot()) == before


def test_semantic_delta_canonicalizes_display_refs_relation_aliases_and_exact_source_text():
    store, task = _store()
    compiler = SemanticDeltaCompiler(store, task_id="delta-task", task_type="numeric_solve")

    compiler.apply(
        "planner",
        {
            "nodes": [
                {
                    "temp_id": "f1",
                    "type": "fact",
                    "content": {"quantity": 2},
                    "grounding": [{"source": "Alice has 2 apples", "quote": "Alice has 2 apples"}],
                },
                {
                    "temp_id": "p1",
                    "type": "plan_step",
                    "content": {"operation": "identity", "inputs": ["f1"]},
                    "grounding": [],
                },
            ],
            "edges": [{"source": "fact@f1", "relation": "input", "target": "plan@p1"}],
            "send_hints": [],
            "complete": True,
        },
        visible_node_ids=[task.node_id],
    )

    assert any(edge.relation == "input_to" for edge in store.snapshot().edges)


def test_candidate_answer_gets_runtime_result_boundary():
    store, task = _store()
    compiler = SemanticDeltaCompiler(store, task_id="delta-task", task_type="numeric_solve")
    compiler.apply(
        "solver",
        {
            "nodes": [
                {"temp_id": "calc", "type": "calculation", "content": {"expression": "2+3", "value": 5}, "grounding": []},
                {"temp_id": "answer", "type": "candidate_answer", "content": {"value": 5}, "grounding": []},
            ],
            "edges": [{"source": "calc", "relation": "produces", "target": "answer"}],
            "send_hints": [],
            "complete": True,
        },
        visible_node_ids=[task.node_id],
    )
    result = store.latest_valid("delta-task", "main", "result")
    assert result is not None
    assert result.provenance["origin"] == NodeOrigin.RUNTIME
    # Runtime result boundaries expose the canonical domain value, not the
    # candidate node's wrapper object.
    assert result.content["value"] == 5


def test_evaluator_only_nodes_are_not_visible_to_model_roles():
    store, _ = _store()
    hidden = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="gold",
        node_type="gold_label",
        content="5",
        owner="evaluator",
        created_by_role="evaluator",
        provenance={
            "origin": NodeOrigin.EVALUATION,
            "visibility_class": VisibilityClass.EVALUATOR_ONLY,
        },
    )
    assert not node_visible_to_role(hidden, "planner")
    assert not node_visible_to_role(hidden, "solver")
    assert not node_visible_to_role(hidden, "tool")


def test_protocol_accepts_typed_semantics_and_rejects_internal_delta_shape():
    action = {"op": "plan", "facts": [{"content": "there are 2 apples", "source": "task", "quote": "2 apples"}], "strategy": "add"}
    assert validate_action("planner", action, task_type="numeric_solve", protocol_mode="semantic_delta_v2") == []
    internal = {"op": "propose_delta", "nodes": [], "edges": [], "send_hints": [], "complete": True}
    assert validate_action("planner", internal, task_type="numeric_solve", protocol_mode="semantic_delta_v2")
    assert validate_action("critic", action, task_type="numeric_solve")


def test_v2_planner_rejects_non_string_fact_content():
    action = {
        "op": "plan",
        "facts": [{"content": {"quantity": 2}, "source": "task", "quote": "2 apples"}],
        "strategy": "add",
    }
    errors = validate_action(
        "planner",
        action,
        task_type="numeric_solve",
        protocol_mode="semantic_delta_v2",
    )
    assert errors == ["plan.facts[0].content must be a non-empty string"]


def test_v2_solver_rejects_removed_numeric_and_choice_shapes():
    assert validate_action(
        "solver",
        {"op": "solve_numeric", "expression": "2+3", "value": 5},
        task_type="numeric_solve",
        protocol_mode="semantic_delta_v2",
    ) == ["solve_numeric.value must be a non-empty string"]
    errors = validate_action(
        "solver",
        {
            "op": "choose_option",
            "answer": "A",
            "slot_analysis": [],
            "eliminated_options": {},
            "confidence": 0.8,
        },
        task_type="multiple_choice",
        protocol_mode="semantic_delta_v2",
    )
    assert "choose_option.slot_reasons must be a non-empty array" in errors
    assert "choose_option.eliminated_options must be a non-empty object" in errors


def test_v2_critic_exposes_only_verdict_and_diagnosis_and_has_no_model_finalizer():
    judgment = {
        "op": "judge_numeric", "model_verdict": "inconsistent", "error_type": "arithmetic",
        "error_location": "expression", "reason": "sum is wrong",
        "repair_instruction": "recompute the sum",
    }
    assert validate_action("critic", judgment, task_type="numeric_solve", protocol_mode="semantic_delta_v2") == []
    assert validate_action(
        "critic", {**judgment, "target": "result"},
        task_type="numeric_solve", protocol_mode="semantic_delta_v2",
    )
    assert validate_action(
        "final_solver", {"op": "answer", "source": "result", "value": 5},
        task_type="numeric_solve", protocol_mode="semantic_delta_v2",
    )


def test_v2_judgment_kernel_selects_target_and_records_semantic_invalidation():
    store, task = _store()
    workflow = LangGraphWorkflow(
        store=store, model=lambda request: {}, task_id="delta-task",
        task_type="numeric_solve", action_protocol="semantic_delta_v2",
    )
    workflow.compiler.set_node_ref_context({"visible_node_ids": [task.node_id]})
    workflow.compiler.apply("solver", {"op": "solve_numeric", "expression": "2+3", "value": "6"})
    result = store.latest_valid("delta-task", "main", "result")
    assert result is not None
    workflow.compiler.apply("critic", {
        "op": "judge_numeric", "model_verdict": "inconsistent", "error_type": "arithmetic",
        "error_location": "expression", "reason": "2+3 is not 6",
        "repair_instruction": "recompute",
    })
    verification = store.latest_valid("delta-task", "main", "verification")
    assert verification is not None and verification.content["target"] == "result"
    assert store.node(result.node_id).validation["model_judged_incorrect"] is True
    assert any(
        edge.source == verification.node_id and edge.target == result.node_id and edge.relation == "contradicts"
        for edge in store.snapshot().edges
    )


def test_complete_delta_requires_task_spec_outputs_and_is_atomic():
    store, task = _store()
    compiler = SemanticDeltaCompiler(store, task_id="delta-task", task_type="numeric_solve")
    before = graph_state_to_dict(store.snapshot())
    with pytest.raises(ValueError, match="missing required planner output"):
        compiler.apply(
            "planner",
            {"nodes": [], "edges": [], "send_hints": [], "complete": True},
            visible_node_ids=[task.node_id],
        )
    assert graph_state_to_dict(store.snapshot()) == before


def test_dynamic_grant_rejects_evaluator_and_tool_only_nodes():
    store, _ = _store()
    evaluator = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="gold",
        node_type="gold_label",
        content="secret",
        owner="evaluator",
        created_by_role="evaluator",
        provenance={"origin": NodeOrigin.EVALUATION, "visibility_class": VisibilityClass.EVALUATOR_ONLY},
    )
    tool_only = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="hidden_test",
        node_type="official_test_source",
        content="assert secret()",
        owner="tool",
        created_by_role="tool",
        provenance={"origin": NodeOrigin.RUNTIME, "visibility_class": VisibilityClass.TOOL_ONLY},
    )
    views = AgentGraphViewManager(store, task_id="delta-task", branch_id="main", agents=("solver", "tool"))
    with pytest.raises(ValueError, match="restricted"):
        views.grant("solver", node_ids=[evaluator.node_id])
    with pytest.raises(ValueError, match="restricted"):
        views.grant("solver", node_ids=[tool_only.node_id])
    views.grant("tool", node_ids=[tool_only.node_id])
    assert tool_only.node_id in views.view("tool").visible_node_ids


def test_private_code_verification_never_echoes_hidden_assertion():
    secret = "8675309"
    result = CodeVerifier().verify(
        "def secret_fn():\n    return 0",
        private_tests=[f"assert secret_fn() == {secret}"],
        evaluation=EvaluationContext(task_id="delta-task", private_tests=(secret,)),
    )
    rendered = str(result.as_dict())
    assert not result.passed
    assert secret not in rendered
    assert "assert secret_fn" not in rendered
    assert "private verification" in result.failure_summary


def test_existing_send_hint_must_be_in_sender_view():
    store, task = _store()
    hidden_from_sender = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="other_source",
        node_type="task_source",
        content="other",
        owner="dataset",
        created_by_role="dataset",
        provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC},
    )
    compiler = SemanticDeltaCompiler(store, task_id="delta-task", task_type="numeric_solve")
    with pytest.raises(SemanticDeltaValidationError, match="not visible to sender"):
        compiler.apply(
            "planner",
            {
                "nodes": [],
                "edges": [],
                "send_hints": [{"receiver": "solver", "fragments": [hidden_from_sender.node_id], "reason": "x"}],
                "complete": False,
            },
            visible_node_ids=[task.node_id],
        )


def test_protocol_modes_are_strictly_isolated():
    delta = {"op": "propose_delta", "nodes": [], "edges": [], "send_hints": [], "complete": False}
    legacy = {"op": "add_fact", "id": "A", "value": 2}
    typed = {"op": "plan", "facts": [{"content": "the stated quantity is 2", "source": "task", "quote": "2"}], "strategy": "add"}
    assert validate_action("planner", delta, task_type="numeric_solve", protocol_mode="legacy")
    assert validate_action("planner", legacy, task_type="numeric_solve", protocol_mode="semantic_delta_v2")
    assert validate_action("planner", delta, task_type="table_qa", protocol_mode="semantic_delta_v2")
    assert validate_action("planner", typed, task_type="numeric_solve", protocol_mode="semantic_delta_v2") == []
    assert validate_action("planner", typed, task_type="numeric_solve", protocol_mode="legacy")


def test_semantic_commit_preserves_branch_ancestry():
    store, _ = _store()
    store.create_branch("delta-task", "candidate")
    branch_task = store.latest_valid("delta-task", "candidate", "task")
    assert branch_task is not None
    compiler = SemanticDeltaCompiler(
        store,
        task_id="delta-task",
        branch_id="candidate",
        task_type="numeric_solve",
    )
    compiler.apply(
        "planner",
        {
            "nodes": [
                {"temp_id": "f1", "type": "fact", "content": {"quantity": 2}, "grounding": [{"source": branch_task.node_id, "quote": "2"}]},
                {"temp_id": "p1", "type": "plan_step", "content": {"operation": "identity", "inputs": ["f1"]}, "grounding": []},
            ],
            "edges": [{"source": "f1", "relation": "input_to", "target": "p1"}],
            "send_hints": [],
            "complete": True,
        },
        visible_node_ids=[branch_task.node_id],
    )
    assert store.branch_info("delta-task", "candidate")["parent_branch_id"] == "main"


def test_repair_requires_post_verification_semantic_output():
    store, task = _store()
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="delta-task",
        task_type="numeric_solve",
        action_protocol="semantic_delta_v2",
    )
    workflow.compiler.set_node_ref_context({"visible_node_ids": [task.node_id]})
    answer = {"op": "solve_numeric", "expression": "2+3", "value": "5"}
    workflow.compiler.apply("solver", answer)
    store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"status": "need_fix"},
        owner="critic",
        status="need_fix",
        created_by_role="critic",
    )
    assert any("post-verification" in item for item in workflow._completion_errors("solver", mode="repair"))
    answer = {"op": "solve_numeric", "expression": "3+3", "value": "6"}
    workflow.compiler.apply("solver", answer)
    assert workflow._completion_errors("solver", mode="repair") == []


def test_v2_non_hotpot_repair_prompt_requires_typed_semantics():
    store, _ = _store()
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="delta-task",
        task_type="numeric_solve",
        action_protocol="semantic_delta_v2",
    )
    suffix = workflow._action_role_suffix("solver", "repair")
    assert '"op":"solve_numeric"' in suffix
    assert "Do not emit graph structure" in suffix
    assert "temporary IDs" not in suffix


def test_v2_multiple_choice_critic_uses_independent_audit_prompt():
    store = GraphStore()
    store.add_node(
        task_id="choice-prompt",
        branch_id="main",
        logical_id="task",
        node_type="task_source",
        content="Which number equals two plus two?",
        owner="dataset",
        created_by_role="dataset",
        provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC},
    )
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="choice-prompt",
        task_type="multiple_choice",
        action_protocol="semantic_delta_v2",
    )
    suffix = workflow._action_role_suffix("critic", "normal")
    assert "INDEPENDENT multiple-choice auditor" in suffix
    assert "never accept them by default" in suffix
    assert "EVERY manifest slot_claim" in suffix
    assert "claim_id already identifies the option label or slot" in suffix
    assert "concrete option text" in suffix
    assert "The selected option matches the question." not in suffix


def test_v2_choice_reason_error_is_rendered_back_to_critic():
    store = GraphStore()
    common = {
        "task_id": "choice-feedback", "branch_id": "main",
        "owner": "dataset", "created_by_role": "dataset",
        "provenance": {"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC},
    }
    store.add_node(logical_id="task", node_type="task_source", content="Choose the safe practice.", **common)
    store.add_node(logical_id="choice_schema", node_type="choice_schema", content={"slot_values": {"A": ["Safe practice"], "B": ["Unsafe practice"]}}, **common)
    store.add_node(logical_id="choice_source_A", node_type="choice_source", content={"label": "A", "text": "Safe practice"}, **common)
    store.add_node(logical_id="choice_source_B", node_type="choice_source", content={"label": "B", "text": "Unsafe practice"}, **common)
    store.add_node(
        logical_id="result", node_type="result", content={"answer": "A", "slot_analysis": [{"slot": 1, "selected_term": "Safe practice", "reason": "slot 1 Safe practice"}], "option_reviews": [{"label": "A", "assessment": "selected", "reason": "A Safe practice"}, {"label": "B", "assessment": "rejected", "reason": "B Unsafe practice"}]}, **common,
    )
    store.add_node(
        logical_id="verification", node_type="verification", status="uncertain",
        content={"error_type": "invalid_choice_audit_reason", "reason": '{"repair_packet":{"invalid_claims":[{"claim_id":"result@v1:elimination:A","missing_references":["option_label","option_text"]}]}}'},
        task_id="choice-feedback", branch_id="main", owner="critic", created_by_role="critic",
    )
    workflow = LangGraphWorkflow(store=store, model=lambda request: {}, task_id="choice-feedback", task_type="multiple_choice", action_protocol="semantic_delta_v2")
    rendered = workflow._append_choice_audit_manifest("<GRAPH>", "critic")
    assert "CRITIC REPAIR PACKET" in rendered
    assert "result@v1:elimination:A" in rendered


def test_choice_reason_uses_claim_id_for_identity_not_repeated_number_or_label():
    store = GraphStore()
    common = {"task_id": "choice-identity", "branch_id": "main", "owner": "dataset", "created_by_role": "dataset"}
    store.add_node(logical_id="task", node_type="task_source", content="Choose the safe practice.", **common)
    store.add_node(logical_id="choice_schema", node_type="choice_schema", content={"allowed_labels": ["A", "B"], "slot_values": {"A": ["Safe practice"], "B": ["Unsafe practice"]}}, **common)
    store.add_node(logical_id="choice_source_A", node_type="choice_source", content={"label": "A", "text": "Safe practice"}, **common)
    store.add_node(logical_id="choice_source_B", node_type="choice_source", content={"label": "B", "text": "Unsafe practice"}, **common)
    compiler = ActionCompiler(store, task_id="choice-identity", task_type="multiple_choice")
    compiler.apply("solver", {
        "op": "choose_option", "answer": "A",
        "slot_reasons": [{"slot": 1, "reason": "Safe practice appears to fit"}],
        "option_reviews": [
            {"label": "A", "assessment": "selected", "reason": "Safe practice appears to fit"},
            {"label": "B", "assessment": "rejected", "reason": "Unsafe practice appears not to fit"},
        ],
        "confidence": 0.5,
    })
    manifest = build_choice_audit_manifest(store, task_id="choice-identity")
    assert manifest is not None
    action = {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"],
        "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [{
            "claim_id": manifest["slot_claims"][0]["claim_id"],
            "verdict": "contradicted",
            "reason": "The selected term 'Safe practice' conflicts in the first blank.",
        }],
        "option_checks": [
            {
                "claim_id": item["claim_id"],
                "verdict": "contradicted" if item["option"] == "A" else "supported",
                "reason": f"The concrete text '{item['option_text']}' was checked against the question.",
            }
            for item in manifest["elimination_claims"]
        ],
        "selected_option_check": {
            "verdict": "contradicted",
            "reason": "The selected text 'Safe practice' conflicts with the question.",
        },
        "summary": "The selected claim is contradicted.",
    }
    verification = compiler.critic_verification_action(action)
    assert verification["status"] == "need_fix"
    assert verification["error_type"] != "invalid_choice_audit_reason"


def test_choice_audit_rejects_inconsistent_solver_critic_verdict():
    store = GraphStore()
    common = {"task_id": "choice-consistency", "branch_id": "main", "owner": "dataset", "created_by_role": "dataset"}
    store.add_node(logical_id="task", node_type="task_source", content="Choose the safe practice.", **common)
    store.add_node(logical_id="choice_source_A", node_type="choice_source", content={"label": "A", "text": "Safe practice"}, **common)
    store.add_node(logical_id="choice_source_B", node_type="choice_source", content={"label": "B", "text": "Unsafe practice"}, **common)
    compiler = ActionCompiler(store, task_id="choice-consistency", task_type="multiple_choice")
    compiler.apply("solver", {"op": "choose_option", "answer": "A", "slot_reasons": [{"slot": 1, "reason": "Safe practice"}], "option_reviews": [{"label": "A", "assessment": "selected", "reason": "Safe practice"}, {"label": "B", "assessment": "rejected", "reason": "Unsafe practice"}], "confidence": 1})
    manifest = build_choice_audit_manifest(store, task_id="choice-consistency")
    assert manifest is not None
    action = {
        "op": "audit_choice", "candidate_version": manifest["candidate_version"], "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [{"claim_id": "unused", "verdict": "insufficient", "reason": "x"}],
        "option_checks": [
            {"claim_id": item["claim_id"], "verdict": "contradicted" if item["option"] == "A" else "supported", "reason": item["option_text"]}
            for item in manifest["elimination_claims"]
        ],
        "selected_option_check": {"verdict": "supported", "reason": "Safe practice"},
        "summary": "audit",
    }
    result = compiler.critic_verification_action(action)
    assert result["error_type"] in {"invalid_choice_audit_scope", "invalid_choice_audit_reason"}


def test_v2_hotpot_typed_repair_requires_a_new_result_not_old_boundary():
    store = GraphStore()
    question = store.add_node(
        task_id="hotpot-v2",
        branch_id="main",
        logical_id="task",
        node_type="question_source",
        content="Where was the bridge born?",
        owner="dataset",
        created_by_role="dataset",
        provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC},
    )
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="hotpot-v2",
        task_type="multihop_qa",
        action_protocol="semantic_delta_v2",
    )
    sentence_ids = []
    for index, text in enumerate(("The bridge was born in London.", "London is in England."), start=1):
        sentence = store.add_node(
            task_id="hotpot-v2", branch_id="main", logical_id=f"sentence_source_{index}",
            node_type="sentence_source", content={"text": text}, owner="dataset",
            created_by_role="dataset",
            provenance={"origin": NodeOrigin.SOURCE, "visibility_class": VisibilityClass.PUBLIC},
        )
        sentence_ids.append(sentence.node_id)
    workflow.agent_views.grant_global_visibility()
    workflow.compiler.set_node_ref_context({"visible_node_ids": [question.node_id, *sentence_ids]})
    workflow.compiler.apply("solver", {
        "op": "answer_with_evidence", "answer": "London", "bridge_entity": "X",
        "citations": ["sentence_source_1", "sentence_source_2"],
        "reasoning": ["bridge born in London", "London is in England"],
    })
    store.add_node(
        task_id="hotpot-v2", branch_id="main", logical_id="verification",
        node_type="verification", content={"status": "need_fix"}, owner="critic",
        status="need_fix", created_by_role="critic",
    )
    assert workflow._completion_errors("solver", mode="repair")
    workflow.compiler.apply("solver", {
        "op": "repair_patch",
        "action": "REVISE_ANSWER",
        "patch": {"answer": "Paris", "reasoning_chain": ["s1 -> Paris"]},
    })
    assert workflow._completion_errors("solver", mode="repair") == []


def test_model_selected_enforces_receiver_relative_closure_budget():
    store, _ = _store()
    optional = store.add_node(
        task_id="delta-task",
        branch_id="main",
        logical_id="optional_note",
        node_type="critique",
        content={"detail": "many optional tokens that exceed a zero budget"},
        owner="planner",
        created_by_role="planner",
        provenance={
            "origin": NodeOrigin.MODEL,
            "visibility_class": VisibilityClass.ROLE_SCOPED,
            "communication_scope": "OPTIONAL_POLICY",
        },
    )
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="delta-task",
        task_type="numeric_solve",
        communication_policy=make_communication_policy("model_selected"),
        communication_budget_tokens=0,
        action_protocol="semantic_delta_v2",
    )
    hint = fragments_for_node(optional, node_scope="OPTIONAL_POLICY")[0].fragment_id
    event = workflow._communicate_nodes(
        sender="planner",
        node_ids=[optional.node_id],
        branch_id="main",
        model_send_hints={"solver": [hint]},
    )[0]
    assert hint in event["kernel_rejected_fragment_ids"]
    assert optional.node_id not in workflow.agent_views.view("solver").visible_node_ids
