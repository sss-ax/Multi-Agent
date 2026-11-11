from baselines.aflow_artifact_graph import artifact_records, classify_output
from baselines.aflow_artifact_router import route_receiver_context
from baselines.aflow_graph_adapter import AFlowGraphSession
from baselines.aflow_profiles import ReceiverContract
from workflow_runtime.communication import SendAllPolicy


def test_unknown_output_is_opaque_and_digest_is_stable():
    first = artifact_records("free text", operator="Agent", invocation_id="i")
    second = artifact_records("free text", operator="Agent", invocation_id="i")
    assert classify_output("free text") == ("opaque_text",)
    assert first[0].artifact_id == second[0].artifact_id
    assert first[0].original_payload_ref == ""


def test_structured_answer_becomes_candidate_artifact():
    records = artifact_records({"answer": "Paris", "confidence": 0.9}, operator="AnswerGenerate", invocation_id="i")
    assert {item.artifact_type for item in records} == {"candidate_answer", "uncertainty"}


def test_artifact_mode_preserves_candidate_multiplicity_and_routes():
    session = AFlowGraphSession(
        task_id="artifact-test", task="q", dataset="HotpotQA", workflow_round=3,
        profile_explicit=True, context_mode="artifact_minctx", content_dedup=True,
        invocation_reuse=True, policy=SendAllPolicy(),
        task_metadata={"question": "q", "context": []},
    )
    session.compile_output(operator="AnswerGenerate", output={"answer": "Paris"}, input_kwargs={"input": "q"})
    session.compile_output(operator="AnswerGenerate", output={"answer": "Paris"}, input_kwargs={"input": "q"})
    result = route_receiver_context(session, "ScEnsemble", {"solutions": ["Paris", "Paris"]},
        ReceiverContract("ScEnsemble", ("candidate_answer",), preserve_multiplicity=True))
    assert result.contract_satisfied
    assert result.kwargs["solutions"] == ["Paris", "Paris"]
    assert session.telemetry.summary()["aflow_artifact_count"] >= 2


def test_contract_failure_returns_original_kwargs():
    session = AFlowGraphSession(task_id="artifact-empty", task="q", dataset="GSM8K",
                                context_mode="artifact_minctx", profile_explicit=True)
    original = {"input": "unchanged"}
    result = route_receiver_context(session, "Custom", original,
        ReceiverContract("Custom", ("candidate_answer",)))
    assert result.fallback
    assert result.kwargs == original


def test_mbpp_source_producer_without_roots_is_not_fallback():
    session = AFlowGraphSession(
        task_id="mbpp-producer", task="problem", dataset="MBPP", workflow_round=14,
        context_mode="artifact_minctx", profile_explicit=True,
    )
    original = {"problem": "p", "entry_point": "f", "instruction": "generate"}
    result = route_receiver_context(session, "CustomCodeGenerate", original,
        session.profile.receiver_contract("CustomCodeGenerate"))
    assert result.contract_satisfied
    assert not result.fallback
    assert result.kwargs == original


def test_evaluator_fields_are_not_compiled_as_artifacts():
    records = artifact_records({"answer": "Paris", "supporting_facts": [["Paris", 0]], "score": 1.0}, operator="AnswerGenerate", invocation_id="i")
    assert all(item.artifact_type != "evidence_ref" for item in records)
