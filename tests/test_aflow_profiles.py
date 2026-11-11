from __future__ import annotations

import pytest

from baselines.aflow_graph_adapter import AFlowGraphSession, extract_weighted_solution_letter
from baselines.aflow_profiles import get_aflow_profile


def test_profile_registry_marks_only_gsm8k_validated():
    assert get_aflow_profile("GSM8K", 10).validated is True
    assert get_aflow_profile("MATH", 5).validated is False
    assert get_aflow_profile("DROP", 3).validated is True
    assert get_aflow_profile("HotpotQA", 3).validated is True
    assert get_aflow_profile("MBPP", 14).validated is True
    assert get_aflow_profile("HumanEval", 5).validated is False


def _hotpot_session(*, policy: str = "full", budget: int = 0, top_k: int = 4):
    context = [
        ["Doc A", ["Alice works at Acme.", "Acme is based in Paris."]],
        ["Doc B", ["Bob works at Beta.", "Beta is based in Rome."]],
    ]
    question = "Where is Alice's company based?"
    original = "Context: " + "\n".join(" ".join(item[1]) for item in context)
    original += f"\n\nQuestion: {question}\n\nAnswer:"
    session = AFlowGraphSession(
        task_id="hotpot-source",
        task=question,
        dataset="HotpotQA",
        workflow_round=3,
        profile_explicit=True,
        context_mode="graph_minctx",
        context_policy=policy,
        context_budget_tokens=budget,
        context_top_k_documents=top_k,
        task_metadata={"question": question, "context": context},
    )
    return session, original


def test_hotpot_source_graph_roundtrip_and_no_gold_labels():
    session, original = _hotpot_session()
    graph = session.hotpot_source_graph()
    assert len(graph["question"]) == 1
    assert len(graph["documents"]) == 2
    assert len(graph["sentences"]) == 4
    assert session.render_hotpot_context(graph["sentences"]) == original
    contents = [node.content for node in session.store.snapshot().nodes.values()]
    assert not any("supporting_facts" in str(value) for value in contents)


def test_hotpot_graph_minctx_full_is_lossless():
    session, original = _hotpot_session(policy="full")
    prepared = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": original},
        communication_mode="graph_minctx",
    )
    assert prepared.kwargs["input"] == original
    assert prepared.source_node_ids
    assert session._context_plans[-1].fallback is False
    assert session._context_plans[-1].contract_satisfied is True


def test_hotpot_topk_context_is_shorter_and_deterministic():
    session, original = _hotpot_session(policy="topk_documents", top_k=1)
    prepared = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": original},
        communication_mode="graph_minctx",
    )
    assert len(prepared.kwargs["input"]) < len(original)
    assert len({session.store.node(node_id).provenance["document_index"] for node_id in prepared.source_node_ids}) == 1
    assert session._context_plans[-1].fallback is False


def test_hotpot_dependency_minimal_context_adds_sentence_neighbors():
    session, original = _hotpot_session(policy="dependency_minimal", top_k=1)
    prepared = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": original},
        communication_mode="graph_minctx",
    )
    assert prepared.kwargs["input"]
    assert session._context_plans[-1].fallback is False
    assert session._context_plans[-1].closure_node_ids


def test_hotpot_context_budget_falls_back_to_original_without_visibility_mutation():
    session, original = _hotpot_session(policy="topk_documents", budget=1, top_k=1)
    before = set(session.store.snapshot().nodes)
    prepared = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": original},
        communication_mode="graph_minctx",
    )
    assert prepared.kwargs["input"] == original
    assert session._context_plans[-1].fallback is True
    assert session._context_plans[-1].fallback_reason == "context_budget_exceeded"
    assert set(session.store.snapshot().nodes) == before


def test_hotpot_flat_topk_alias_and_selection_audit_are_deterministic():
    left, original = _hotpot_session(policy="topk_documents", top_k=1)
    right, _ = _hotpot_session(policy="flat_topk_documents", top_k=1)
    for session in (left, right):
        session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    assert left._context_plans[-1].selected_node_ids == right._context_plans[-1].selected_node_ids
    plan = left._context_plans[-1]
    assert plan.document_scores and plan.sentence_scores
    assert plan.tokens_by_level and plan.risk_by_level


def test_hotpot_adaptive_recovery_records_levels_and_uses_original_at_l2():
    session, original = _hotpot_session(policy="dependency_minimal", top_k=1)
    session.context_recovery = "adaptive"
    session.context_recovery_top_k_documents = 2
    prepared = session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    plan = session._context_plans[-1]
    assert plan.attempted_levels[0] == "L0"
    assert plan.effective_level in {"L0", "L1", "L2"}
    assert set(plan.tokens_by_level) >= {"L0"}
    if plan.effective_level == "L2":
        assert prepared.kwargs["input"] == original


def test_hotpot_question_without_valid_entity_does_not_trigger_entity_risk():
    question = "What is mentioned?"
    context = [["Doc A", ["A short fact is mentioned."]], ["Doc B", ["Another fact is mentioned."]]]
    session = AFlowGraphSession(
        task_id="no-entity", task=question, dataset="HotpotQA", workflow_round=3,
        profile_explicit=True, context_mode="graph_minctx", context_policy="full",
        task_metadata={"question": question, "context": context},
    )
    selection = session._hotpot_select_source_nodes("full", 4)
    risk = session._hotpot_context_risk(selection, original_tokens=100, selected_tokens=50, level="L0")
    assert "low_entity_coverage" not in risk.soft_reasons
    assert "question_entities_uncovered" not in risk.critical_reasons


def test_hotpot_bridge_detects_shared_sentence_entity():
    context = [
        ["Alice", ["Alice founded Acme."]],
        ["Paris", ["Acme is based in Paris."]],
    ]
    question = "Where is Alice's company based?"
    session = AFlowGraphSession(
        task_id="bridge", task=question, dataset="HotpotQA", workflow_round=3,
        profile_explicit=True, context_mode="graph_minctx", context_policy="dependency_minimal",
        task_metadata={"question": question, "context": context},
    )
    selection = session._hotpot_select_source_nodes("dependency_minimal", 2)
    assert selection.bridge_paths


def test_hotpot_selection_plan_cache_is_reused():
    session, original = _hotpot_session(policy="dependency_minimal", top_k=2)
    session.context_recovery = "adaptive"
    session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    session.prepare_operator_input(operator="AnswerGenerate", kwargs={"input": original}, communication_mode="graph_minctx")
    assert session._context_plan_cache_hits > 0
    assert session._context_plans[-1].cache_hit is True


def test_humaneval_profile_preserves_generation_and_test_dependency():
    session = AFlowGraphSession(
        task_id="he-profile", task="def f(x):", dataset="HumanEval",
        workflow_round=5, profile_explicit=True,
    )
    profile = get_aflow_profile("HumanEval", 5)
    generated = session.compile_output(
        operator="CustomCodeGenerate",
        output={"response": "def f(x): return x"},
        input_kwargs={"problem": "def f(x):", "entry_point": "f", "instruction": ""},
    )

    retry = profile.input_binding(
        session=session,
        operator="CustomCodeGenerate",
        original_kwargs={
            "problem": "def f(x):",
            "entry_point": "f",
            "instruction": "improve",
        },
        delivered_artifacts=(),
    )
    test = profile.input_binding(
        session=session,
        operator="Test",
        original_kwargs={
            "problem": "def f(x):",
            "solution": "ignored",
            "entry_point": "f",
        },
        delivered_artifacts=({"node_id": generated.primary_node_id},),
    )

    assert retry.source_node_ids == ()
    assert retry.kwargs["instruction"] == "improve"
    assert test.source_node_ids == (generated.primary_node_id,)
    assert test.kwargs["solution"] == "def f(x): return x"
    assert profile.allows_reuse("CustomCodeGenerate", {"instruction": ""}) is False


def test_humaneval_explicitly_skips_all_three_optimizations():
    session = AFlowGraphSession(
        task_id="he-skip", task="def f(x):", dataset="HumanEval",
        workflow_round=5, content_dedup=True, invocation_reuse=True,
        profile_explicit=True,
    )
    assert session.content_dedup is False
    assert session.invocation_reuse_enabled is False
    assert session.optimization_status["content_dedup"]["reason"] == "profile_not_supported"
    assert session.optimization_status["invocation_reuse"]["reason"] == "profile_not_supported"
    assert session.profile.ensemble_operators == ()


def test_math_profile_preserves_refine_prompt_and_does_not_feed_detailed():
    session = AFlowGraphSession(task_id="m1", task="x", dataset="MATH", workflow_round=5)
    profile = get_aflow_profile("MATH", 5)
    program = session.compile_output(operator="Programmer", output={"output": "x=2"})

    refine = profile.input_binding(
        session=session,
        operator="Custom",
        original_kwargs={"input": "problem\nCode output: x=2", "instruction": "REFINE_ANSWER_PROMPT"},
        delivered_artifacts=({"node_id": program.primary_node_id},),
    )
    assert refine.kwargs["input"] == "x\nCode output: x=2"

    detailed = profile.input_binding(
        session=session,
        operator="Custom",
        original_kwargs={"input": "x", "instruction": "DETAILED_SOLUTION_PROMPT"},
        delivered_artifacts=({"node_id": program.primary_node_id},),
    )
    assert detailed.kwargs == {"input": "x", "instruction": "DETAILED_SOLUTION_PROMPT"}
    assert detailed.source_node_ids == ()


def test_unregistered_dataset_cannot_take_over_graph_input():
    session = AFlowGraphSession(task_id="u1", task="x", dataset="Unknown", workflow_round=1)
    with pytest.raises(RuntimeError, match="requires a registered profile"):
        session.prepare_operator_input(operator="Custom", kwargs={"input": "x"}, communication_mode="graph_sendall")


def test_random_generation_does_not_allow_profile_invocation_reuse():
    session = AFlowGraphSession(
        task_id="g1", task="x", dataset="GSM8K", workflow_round=10,
        content_dedup=True, invocation_reuse=True, generation_config={"do_sample": True, "temperature": 0.7},
    )
    assert session.can_reuse_operator("Custom") is False


def test_math_reuse_is_limited_to_generate_custom():
    session = AFlowGraphSession(
        task_id="m2", task="x", dataset="MATH", workflow_round=5,
        content_dedup=True, invocation_reuse=True, generation_config={"do_sample": False, "temperature": 0},
    )
    assert session.can_reuse_operator("Custom", {"instruction": "GENERATE_SOLUTION_PROMPT"})
    assert not session.can_reuse_operator("Custom", {"instruction": "REFINE_ANSWER_PROMPT"})
    assert not session.can_reuse_operator("Custom", {"instruction": "DETAILED_SOLUTION_PROMPT"})


def test_math_expanded_prompts_have_exact_roles():
    profile = get_aflow_profile("MATH", 5)
    assert profile.candidate_signature(
        operator="Custom",
        kwargs={"instruction": "Given the mathematical problem and the output from the code execution,\nDetails"},
    ) == "refined"
    assert profile.candidate_signature(
        operator="Custom",
        kwargs={"instruction": "Provide a comprehensive, step-by-step solution to the given mathematical problem.\nDetails"},
    ) == "detailed"
    assert profile.candidate_signature(
        operator="Custom",
        kwargs={"instruction": "Please solve the given mathematical problem step by step.\nDetails"},
    ) == "generated"


def test_math_unknown_instruction_is_not_classified_by_body_words():
    profile = get_aflow_profile("MATH", 5)
    with pytest.raises(ValueError, match="unknown MATH Custom instruction"):
        profile.candidate_signature(
            operator="Custom",
            kwargs={"instruction": "This is a detailed generate prompt"},
        )


def test_math_refine_uses_program_execution_output_not_source_code():
    session = AFlowGraphSession(
        task_id="m-refine", task="solve 2+2", dataset="MATH", workflow_round=5,
        profile_explicit=True,
    )
    programmer = session.compile_output(
        operator="Programmer",
        output={"code": "print(999)", "output": "4"},
    )
    prepared = session.prepare_operator_input(
        operator="Custom",
        kwargs={"input": "ignored", "instruction": "REFINE_ANSWER_PROMPT"},
        communication_mode="graph_sendall",
    )
    assert prepared.source_node_ids == (programmer.primary_node_id,)
    assert prepared.kwargs["input"] == "solve 2+2\nCode output: 4"
    assert "print(999)" not in prepared.kwargs["input"]


def test_math_detailed_and_generate_are_independent_of_programmer():
    session = AFlowGraphSession(
        task_id="m-independent", task="solve x", dataset="MATH", workflow_round=5,
        profile_explicit=True,
    )
    session.compile_output(operator="Programmer", output={"code": "bad", "output": "wrong"})
    detailed = session.prepare_operator_input(
        operator="Custom",
        kwargs={"input": "solve x", "instruction": "DETAILED_SOLUTION_PROMPT"},
        communication_mode="graph_sendall",
    )
    generated = session.prepare_operator_input(
        operator="Custom",
        kwargs={"input": "solve x", "instruction": "GENERATE_SOLUTION_PROMPT"},
        communication_mode="graph_sendall",
    )
    assert detailed.source_node_ids == ()
    assert generated.source_node_ids == ()
    assert detailed.kwargs["input"] == generated.kwargs["input"] == "solve x"


def test_math_role_weighted_candidates_preserve_generated_multiplicity():
    session = AFlowGraphSession(
        task_id="m-candidates", task="solve x", dataset="MATH", workflow_round=5,
        content_dedup=True, profile_explicit=True,
    )
    outputs = (
        ("refined answer", "REFINE_ANSWER_PROMPT"),
        ("detailed answer", "DETAILED_SOLUTION_PROMPT"),
        ("generated answer", "GENERATE_SOLUTION_PROMPT"),
        ("generated answer", "GENERATE_SOLUTION_PROMPT"),
    )
    for value, instruction in outputs:
        session.compile_output(
            operator="Custom",
            output={"response": value},
            input_kwargs={"input": "solve x", "instruction": instruction},
        )
    assert len(session.candidate_pool_ids()) == 3
    multiplicities = [session.artifact_multiplicity(node_id) for node_id in session.candidate_pool_ids()]
    assert multiplicities == [1, 1, 2]
    prompt, mapping = session.weighted_ensemble_prompt({"problem": "solve x"})
    assert list(mapping.values()) == ["refined answer", "detailed answer", "generated answer"]
    assert "[refined] (x1)" in prompt
    assert "[detailed] (x1)" in prompt
    assert "[generated] (x2)" in prompt


def test_math_native_ensemble_expands_multiplicity_before_original_operator():
    session = AFlowGraphSession(
        task_id="m-native-ensemble", task="solve x", dataset="MATH",
        workflow_round=5, content_dedup=True, profile_explicit=True,
    )
    for value, instruction in (
        ("R", "REFINE_ANSWER_PROMPT"),
        ("D", "DETAILED_SOLUTION_PROMPT"),
        ("G", "GENERATE_SOLUTION_PROMPT"),
        ("G", "GENERATE_SOLUTION_PROMPT"),
    ):
        session.compile_output(
            operator="Custom",
            output={"response": value},
            input_kwargs={"input": "solve x", "instruction": instruction},
        )
    assert session.profile.ensemble_strategy == "native_multiplicity"
    assert session.expanded_candidate_values() == ["R", "D", "G", "G"]


def test_math_unknown_custom_instruction_is_rejected():
    profile = get_aflow_profile("MATH", 5)
    session = AFlowGraphSession(task_id="m-unknown", task="x", dataset="MATH", workflow_round=5)
    with pytest.raises(ValueError, match="unknown MATH Custom instruction"):
        profile.input_binding(
            session=session,
            operator="Custom",
            original_kwargs={"input": "x", "instruction": "UNKNOWN"},
            delivered_artifacts=(),
        )


def test_drop_answer_generators_are_independent_and_ensemble_gets_one_candidate():
    session = AFlowGraphSession(
        task_id="d1", task="Question text", dataset="DROP", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    session.compile_output(
        operator="AnswerGenerate", output={"answer": "A"},
        input_kwargs={"input": "Question text"},
    )
    second = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": "Question text"},
        communication_mode="graph_sendall",
    )
    assert second.source_node_ids == ()
    assert second.kwargs == {"input": "Question text"}

    session.compile_output(
        operator="AnswerGenerate", output={"answer": "A"},
        input_kwargs={"input": "Question text"},
    )
    session.compile_output(
        operator="AnswerGenerate", output={"answer": "A"},
        input_kwargs={"input": "Question text"},
    )
    ensemble = session.prepare_operator_input(
        operator="ScEnsemble",
        kwargs={"solutions": ["A", "A", "A"]},
        communication_mode="graph_sendall",
    )
    assert ensemble.kwargs["solutions"] == ["A"]
    assert session.artifact_multiplicity(session.candidate_pool_ids()[0]) == 3
    assert "B:" not in "A: " + str(ensemble.kwargs["solutions"])


def test_drop_mixed_candidates_preserve_occurrence_weights():
    session = AFlowGraphSession(
        task_id="d2", task="Question text", dataset="DROP", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    for value in ("A", "A", "B"):
        session.compile_output(
            operator="AnswerGenerate", output={"answer": value},
            input_kwargs={"input": "Question text"},
        )
    ensemble = session.prepare_operator_input(
        operator="ScEnsemble",
        kwargs={"solutions": ["A", "A", "B"]},
        communication_mode="graph_sendall",
    )
    assert ensemble.kwargs["solutions"] == ["A", "A", "B"]


def test_drop_final_custom_preserves_native_prompt_shape():
    session = AFlowGraphSession(
        task_id="d3", task="Question text", dataset="DROP", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    ensemble = session.compile_output(operator="ScEnsemble", output={"response": "A"})
    prepared = session.prepare_operator_input(
        operator="Custom",
        kwargs={
            "input": "Question: Question text\nBest solution: old",
            "instruction": "REFINE_ANSWER_PROMPT",
        },
        communication_mode="graph_sendall",
    )
    assert prepared.kwargs["input"] == "Question: Question text\nBest solution: A"
    assert prepared.kwargs["instruction"] == "REFINE_ANSWER_PROMPT"


def test_hotpotqa_profile_is_registered_and_answer_only():
    profile = get_aflow_profile("HotpotQA", 3)
    assert profile is not None
    assert profile.validated is True
    assert profile.allows_reuse("AnswerGenerate", {})
    assert not profile.allows_reuse("ScEnsemble", {})
    assert not profile.allows_reuse("Custom", {})


def test_hotpotqa_answer_generators_are_independent_and_ensemble_projects_answer():
    session = AFlowGraphSession(
        task_id="h1", task="Hotpot question", dataset="HotpotQA", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    for value in ("A", "A", "A"):
        session.compile_output(
            operator="AnswerGenerate",
            output={"thought": f"reasoning {value}", "answer": value},
            input_kwargs={"input": "Hotpot question"},
        )

    independent = session.prepare_operator_input(
        operator="AnswerGenerate",
        kwargs={"input": "Hotpot question"},
        communication_mode="graph_sendall",
    )
    assert independent.source_node_ids == ()
    assert independent.kwargs == {"input": "Hotpot question"}

    ensemble = session.prepare_operator_input(
        operator="ScEnsemble",
        kwargs={"solutions": ["A", "A", "A"]},
        communication_mode="graph_sendall",
    )
    assert ensemble.kwargs["solutions"] == ["A"]
    candidate_id = session.candidate_pool_ids()[0]
    assert session.artifact_multiplicity(candidate_id) == 3
    assert session.raw_output(candidate_id)["thought"] == "reasoning A"


def test_hotpotqa_mixed_candidates_preserve_vote_weights():
    session = AFlowGraphSession(
        task_id="h2", task="Hotpot question", dataset="HotpotQA", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    for value in ("A", "A", "B"):
        session.compile_output(
            operator="AnswerGenerate",
            output={"thought": "reasoning", "answer": value},
            input_kwargs={"input": "Hotpot question"},
        )
    ensemble = session.prepare_operator_input(
        operator="ScEnsemble",
        kwargs={"solutions": ["A", "A", "B"]},
        communication_mode="graph_sendall",
    )
    assert ensemble.kwargs["solutions"] == ["A", "A", "B"]


def test_hotpotqa_same_answer_with_different_thoughts_is_one_weighted_candidate():
    session = AFlowGraphSession(
        task_id="h-weight", task="Hotpot question", dataset="HotpotQA", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    for thought in ("first reasoning", "second reasoning", "third reasoning"):
        session.compile_output(
            operator="AnswerGenerate",
            output={"thought": thought, "answer": "Mario Lemieux"},
            input_kwargs={"input": "Hotpot question"},
        )
    assert len(session.candidate_pool_ids()) == 1
    candidate = session.candidate_pool_ids()[0]
    assert session.artifact_multiplicity(candidate) == 3
    assert session.raw_output(candidate)["thought"] == "first reasoning"

    prompt, mapping = session.weighted_ensemble_prompt()
    assert mapping == {"A": "Mario Lemieux"}
    assert "Occurrence count: 3" in prompt
    assert "Allowed candidate IDs: A" in prompt


def test_hotpotqa_weighted_selection_falls_back_to_highest_multiplicity():
    session = AFlowGraphSession(
        task_id="h-fallback", task="Hotpot question", dataset="HotpotQA", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    for value in ("A", "A", "B"):
        session.compile_output(operator="AnswerGenerate", output={"thought": "t", "answer": value})
    _, mapping = session.weighted_ensemble_prompt()
    assert session.weighted_selection({"solution_letter": "N"}, mapping) == "A"
    assert session.telemetry.summary()["aflow_weighted_ensemble_fallbacks"] == 1


def test_hotpotqa_final_custom_uses_best_answer_template():
    session = AFlowGraphSession(
        task_id="h3", task="Hotpot question", dataset="HotpotQA", workflow_round=3,
        content_dedup=True, profile_explicit=True,
    )
    session.compile_output(operator="ScEnsemble", output={"response": "A"})
    prepared = session.prepare_operator_input(
        operator="Custom",
        kwargs={
            "input": "Question: Hotpot question\nBest answer: old",
            "instruction": "FORMAT_ANSWER_PROMPT",
        },
        communication_mode="graph_sendall",
    )
    assert prepared.kwargs["input"] == "Question: Hotpot question\nBest answer: A"
    assert prepared.kwargs["instruction"] == "FORMAT_ANSWER_PROMPT"


def test_mbpp_profile_has_code_specific_dependencies_and_reuse_boundary():
    session = AFlowGraphSession(
        task_id="mbpp-profile", task="write code", dataset="MBPP", workflow_round=14,
        content_dedup=True, invocation_reuse=True,
        generation_config={"do_sample": False, "temperature": 0}, profile_explicit=True,
    )
    profile = get_aflow_profile("MBPP", 14)
    assert profile is not None
    assert profile.validated is True
    assert profile.ensemble_strategy == "weighted_code"
    assert session.can_reuse_operator("CustomCodeGenerate", {})
    assert not session.can_reuse_operator("ScEnsemble", {})
    assert not session.can_reuse_operator("Test", {})
    assert not session.can_reuse_operator("Custom", {})

    prepared = session.prepare_operator_input(
        operator="CustomCodeGenerate",
        kwargs={"problem": "write code", "entry_point": "f", "instruction": "generate"},
        communication_mode="graph_sendall",
    )
    assert prepared.source_node_ids == ()
    assert prepared.kwargs["problem"] == "write code"


def test_weighted_solution_letter_parser_accepts_formatter_text_shapes():
    allowed = ("A", "B")
    assert extract_weighted_solution_letter({"solution_letter": "A"}, allowed) == "A"
    assert extract_weighted_solution_letter('solution_letter = "A"', allowed) == "A"
    assert extract_weighted_solution_letter("<solution_letter>B</solution_letter>", allowed) == "B"
    assert extract_weighted_solution_letter("A", allowed) == "A"
    assert extract_weighted_solution_letter("The answer is A because...", allowed) is None
    assert extract_weighted_solution_letter('solution_letter = "N"', allowed) is None


def test_mbpp_code_candidates_use_exact_response_projection_and_multiplicity():
    session = AFlowGraphSession(
        task_id="mbpp-candidates", task="write code", dataset="MBPP", workflow_round=14,
        content_dedup=True, profile_explicit=True,
    )
    code = "def f(x):\n    return x + 1"
    for _ in range(3):
        session.compile_output(
            operator="CustomCodeGenerate",
            output={"response": code},
            input_kwargs={"problem": "write code", "entry_point": "f", "instruction": "generate"},
        )
    assert len(session.candidate_pool_ids()) == 1
    candidate_id = session.candidate_pool_ids()[0]
    assert session.artifact_multiplicity(candidate_id) == 3
    prompt, mapping = session.weighted_ensemble_prompt({"problem": "write code", "entry_point": "f"})
    assert mapping == {"A": code}
    assert prompt.count(code) == 1
    assert "A (x3):" in prompt


def test_mbpp_weighted_prompt_reads_entry_point_from_task_metadata():
    session = AFlowGraphSession(
        task_id="mbpp-entry", task="write code", dataset="MBPP", workflow_round=14,
        task_metadata={"entry_point": "tuple_to_int"},
        content_dedup=True, profile_explicit=True,
    )
    session.compile_output(operator="CustomCodeGenerate", output={"response": "def tuple_to_int(x): return 1"})
    prompt, _ = session.weighted_ensemble_prompt({"problem": "write code"})
    assert "Entry point: tuple_to_int" in prompt


def test_mbpp_test_binding_delivers_only_selected_code():
    session = AFlowGraphSession(
        task_id="mbpp-test-binding", task="write code", dataset="MBPP", workflow_round=14,
        content_dedup=True, profile_explicit=True,
    )
    code = "def f(x):\n    return x + 1"
    session.compile_output(operator="ScEnsemble", output={"response": code})
    prepared = session.prepare_operator_input(
        operator="Test",
        kwargs={"problem": "write code", "solution": "old", "entry_point": "f"},
        communication_mode="graph_sendall",
    )
    assert prepared.kwargs["solution"] == code
    assert "Occurrence count" not in prepared.kwargs["solution"]


def test_mbpp_code_candidates_do_not_merge_internal_formatting_changes():
    session = AFlowGraphSession(
        task_id="mbpp-exact", task="write code", dataset="MBPP", workflow_round=14,
        content_dedup=True, profile_explicit=True,
    )
    for code in ("def f(x):\n    return x + 1", "def f(x):\n return x+1"):
        session.compile_output(operator="CustomCodeGenerate", output={"response": code})
    assert len(session.candidate_pool_ids()) == 2


def test_mbpp_weighted_selection_falls_back_to_highest_code_multiplicity():
    session = AFlowGraphSession(
        task_id="mbpp-fallback", task="write code", dataset="MBPP", workflow_round=14,
        content_dedup=True, profile_explicit=True,
    )
    for code in ("A", "A", "B"):
        session.compile_output(operator="CustomCodeGenerate", output={"response": code})
    _, mapping = session.weighted_ensemble_prompt({"problem": "write code", "entry_point": "f"})
    assert session.weighted_selection({"solution_letter": "N"}, mapping) == "A"
    assert session.telemetry.summary()["aflow_weighted_ensemble_fallbacks"] == 1

