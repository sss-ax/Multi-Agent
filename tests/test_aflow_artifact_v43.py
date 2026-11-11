from baselines.aflow_graph_adapter import AFlowGraphSession
from baselines.aflow_artifact_graph import CandidateBatchState


def _session():
    return AFlowGraphSession(
        task_id="v43", task="problem", dataset="MBPP", workflow_round=14,
        profile_explicit=True, context_mode="artifact_minctx", content_dedup=True,
    )


def test_mbpp_sources_are_graphstore_nodes():
    session = _session()
    reserved = session.reserve_invocation("CustomCodeGenerate")
    session.build_native_invocation_plan(
        reserved, "CustomCodeGenerate", {"problem": "p", "entry_point": "f"}
    )
    registry = session._mbpp_source_registry
    assert registry is not None
    assert session.store.node(registry.problem_source_id).type == "problem_source"
    assert session.store.node(registry.entry_point_source_id).type == "entry_point_source"


def test_candidate_batch_freeze_does_not_change_occurrences():
    session = _session()
    reserved = session.reserve_invocation("CustomCodeGenerate")
    batch = session.ensure_candidate_batch(reserved)
    assert batch.state == CandidateBatchState.OPEN
    frozen = session.freeze_candidate_batch()
    assert frozen is not None
    assert frozen.state == CandidateBatchState.FROZEN
    assert frozen.occurrence_ids == batch.occurrence_ids


def test_source_producer_does_not_reuse_closed_batch():
    session = _session()
    first = session.reserve_invocation("CustomCodeGenerate")
    first_batch = session.ensure_candidate_batch(first)
    session.freeze_candidate_batch()
    second = session.reserve_invocation("CustomCodeGenerate")
    second_batch = session.ensure_candidate_batch(second)
    assert second_batch.batch_id != first_batch.batch_id
    assert session._active_candidate_batch_id == second_batch.batch_id


def test_occurrence_and_source_payload_domains_are_separate():
    session = _session()
    reserved = session.reserve_invocation("CustomCodeGenerate")
    session.ensure_candidate_batch(reserved)
    plan = session.build_native_invocation_plan(
        reserved, "CustomCodeGenerate", {"problem": "p", "entry_point": "f"}
    )
    session._native_invocation_plans[reserved.invocation_id] = plan
    session.compile_output(
        operator="CustomCodeGenerate", output={"response": "def f(): return 1"},
        input_kwargs={"problem": "p", "entry_point": "f"},
        candidate_batch_id=session._active_candidate_batch_id,
        invocation_index=reserved.invocation_index,
        runtime_invocation_id=reserved.invocation_id,
    )
    occurrence_id = next(iter(session._artifact_occurrences))
    assert session.occurrence_payload(occurrence_id)
    assert session.canonical_for_occurrence(occurrence_id)
    assert session.source_payload(plan.field_bindings["problem"].source_node_ids[0]) == "p"
