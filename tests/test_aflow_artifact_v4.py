from baselines.aflow_graph_adapter import AFlowGraphSession
from baselines.aflow_artifact_router import resolve_ensemble_selection, replay_receiver_closure
from baselines.aflow_profiles import ReceiverContract


def _session():
    return AFlowGraphSession(
        task_id="v4", task="problem", dataset="MBPP", workflow_round=14,
        profile_explicit=True, context_mode="artifact_minctx", content_dedup=True,
    )


def test_runtime_uses_aflow_canonical_nodes_without_agent_artifact_nodes():
    session = _session()
    compiled = session.compile_output(
        operator="CustomCodeGenerate",
        output={"response": "def f(): return 1"},
        input_kwargs={"problem": "problem", "entry_point": "f"},
    )
    nodes = session.store.snapshot().nodes
    assert compiled.primary_node_id in nodes
    assert all(node.type != "agent_artifact" for node in nodes.values())
    assert session.artifact_node_ids()


def test_occurrence_points_to_aflow_canonical_node():
    session = _session()
    compiled = session.compile_output(
        operator="CustomCodeGenerate",
        output={"response": "def f(): return 1"},
        input_kwargs={"problem": "problem", "entry_point": "f"},
    )
    assert all(
        occurrence.canonical_node_id == compiled.primary_node_id
        for occurrence in session._artifact_occurrences.values()
    )


def test_replay_requires_a_real_closure():
    session = _session()
    contract = ReceiverContract("Test", ("code",))
    replay = replay_receiver_closure(
        "Test", None, contract,
        {"problem": "p", "solution": "code"},
        {"problem": "p", "solution": "code"},
    )
    assert replay.replayable is False


def test_invocation_plan_is_created_by_proxy_path_contract():
    session = _session()
    reserved = session.reserve_invocation("CustomCodeGenerate")
    assert reserved.invocation_index == 1
    assert reserved.invocation_id == "CustomCodeGenerate:1"
