from baselines.aflow_graph_adapter import AFlowGraphSession
from baselines.aflow_artifact_graph import BindingValidation, ClosureValidation
from baselines.aflow_artifact_router import replay_receiver_closure
from baselines.aflow_profiles import ReceiverContract


def _session():
    return AFlowGraphSession(
        task_id="v42", task="problem", dataset="MBPP", workflow_round=14,
        profile_explicit=True, context_mode="artifact_minctx", content_dedup=True,
    )


def test_required_binding_missing_is_counted():
    session = _session()
    reserved = session.reserve_invocation("Test")
    plan = session.build_native_invocation_plan(
        reserved, "Test", {"problem": "p", "entry_point": "f"}
    )
    validation = session.validate_required_bindings(plan)
    assert validation.required_field_count >= 3
    assert validation.valid is False
    assert validation.invalid_required_field_count >= 1


def test_raw_nonbinding_is_not_required_binding():
    session = _session()
    reserved = session.reserve_invocation("CustomCodeGenerate")
    plan = session.build_native_invocation_plan(
        reserved, "CustomCodeGenerate", {"problem": "p", "entry_point": "f", "instruction": "raw"}
    )
    validation = session.validate_required_bindings(plan)
    assert validation.required_field_count == 0
    assert validation.valid is True


def test_replay_cannot_succeed_without_binding_validation():
    closure = ClosureValidation(1, 1, 1, 1, complete=True)
    replay = replay_receiver_closure(
        "Test", closure, ReceiverContract("Test", ("code",)),
        {"problem": "p", "solution": "code"},
        {"problem": "p", "solution": "code"},
        binding_valid=False,
    )
    assert replay.replayable is False


def test_replay_requires_complete_closure_and_binding():
    closure = ClosureValidation(1, 1, 1, 1, complete=True)
    replay = replay_receiver_closure(
        "Test", closure, ReceiverContract("Test", ("code",)),
        {"problem": "p", "solution": "code"},
        {"problem": "p", "solution": "code"},
        binding_valid=True,
    )
    assert replay.replayable is True
