from workflow_runtime import (
    CommunicationAction,
    CommunicationActionExecutor,
    CommunicationActionType,
    CommunicationPreview,
    CommunicationReplayBuffer,
    CounterfactualReplayRunner,
    GraphStore,
    LangGraphWorkflow,
)
from workflow_runtime.learning.error_features import (
    assert_deployable_features,
    build_dual_risk_example,
)
from workflow_runtime.learning.schemas import (
    CounterfactualOutcome,
    CurrentAnswerLabel,
    FinalOutcomeLabel,
)
from workflow_runtime.learning.teacher_search import (
    InformationAcquisitionTeacher,
    SendCounterfactualTeacher,
)
from workflow_runtime.learning.risk_model import (
    build_dual_head_network,
    masked_dual_head_bce_loss,
)
from workflow_runtime.learning.artifacts import (
    Phase2ArtifactWriter,
    assert_task_disjoint_splits,
    load_jsonl,
)


def _decision_snapshot(*, budget=256):
    store = GraphStore()
    store.add_node(
        task_id="phase2",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="solve",
        owner="user",
        created_by_role="user",
    )
    plan = store.add_node(
        task_id="phase2",
        branch_id="main",
        logical_id="plan_steps",
        node_type="plan_steps",
        content=[
            {"id": "R1", "operation": "first"},
            {"id": "R2", "operation": "second"},
        ],
        owner="planner",
        created_by_role="planner",
    )
    replay = CommunicationReplayBuffer()
    workflow = LangGraphWorkflow(
        store=store,
        model=lambda request: {},
        task_id="phase2",
        communication_budget_tokens=budget,
        communication_replay_buffer=replay,
    )
    workflow.agent_views.grant("planner", node_ids=[plan.node_id], local=True)
    workflow._communicate_nodes(
        sender="planner",
        node_ids=[plan.node_id],
        branch_id="main",
        workflow_state={"status": "running", "round_id": 0},
    )
    return replay.snapshots()[0], workflow, plan


def _outcome(action, *, current=False, success=False, cost=10, legal=True):
    return CounterfactualOutcome(
        action=action,
        legal=legal,
        current_answer=CurrentAnswerLabel(
            correct=current,
            candidate_node_id="candidate-v1",
            label_source="candidate_oracle",
            error_type="wrong_value" if current is False else "",
        ),
        final_outcome=FinalOutcomeLabel(
            success=success,
            policy_name="test-policy",
            evaluator_source="task_evaluator",
        ),
        communication_tokens=cost,
    )


def test_send_preview_auto_includes_mandatory_fragments_and_applies_to_receiver():
    snapshot, workflow, plan = _decision_snapshot()
    optional = tuple(
        item["fragment_id"] for item in snapshot.fragments
        if item["scope"] == "OPTIONAL_POLICY"
    )
    executor = CommunicationActionExecutor()
    action = CommunicationAction(
        CommunicationActionType.SEND,
        selected_fragment_ids=optional,
    )

    preview = executor.preview(snapshot, action)

    assert preview.legal
    assert plan.node_id in preview.closure_node_ids
    assert any(fragment.endswith("#next_action") for fragment in preview.delivered_fragment_ids)
    assert set(optional).issubset(preview.delivered_fragment_ids)
    workflow.restore_communication_snapshot(snapshot)
    executor.apply(snapshot, action, store=workflow.store, views=workflow.agent_views)
    assert plan.node_id in workflow.agent_views.view("solver").visible_node_ids


def test_send_preview_enforces_budget_after_mandatory_and_closure_rendering():
    snapshot, _, _ = _decision_snapshot(budget=0)

    preview = CommunicationActionExecutor().preview(
        snapshot,
        CommunicationAction(CommunicationActionType.SEND),
    )

    assert not preview.legal
    assert preview.communication_tokens > 0
    assert "exceeds budget" in preview.reason


def test_active_information_action_requires_environment_adapter():
    snapshot, _, _ = _decision_snapshot()

    preview = CommunicationActionExecutor().preview(
        snapshot,
        CommunicationAction(
            CommunicationActionType.RETRIEVE,
            query="missing second-hop evidence",
        ),
    )

    assert not preview.legal
    assert "InformationActionAdapter" in preview.reason


def test_stop_cannot_suppress_protocol_mandatory_fragments():
    snapshot, _, _ = _decision_snapshot()

    preview = CommunicationActionExecutor().preview(
        snapshot,
        CommunicationAction(CommunicationActionType.STOP),
    )

    assert not preview.legal
    assert "mandatory" in preview.reason


def test_active_information_action_rejects_oracle_payload():
    snapshot, _, _ = _decision_snapshot()

    preview = CommunicationActionExecutor().preview(
        snapshot,
        CommunicationAction(
            CommunicationActionType.RETRIEVE,
            query="second hop",
            payload={"gold_answer": "leak"},
        ),
    )

    assert not preview.legal
    assert "oracle-only" in preview.reason


def test_active_information_adapter_can_apply_real_retrieval_mutation():
    snapshot, workflow, _ = _decision_snapshot()

    class Adapter:
        def preview(self, current_snapshot, action):
            return CommunicationPreview(action=action, legal=True, reason="retriever available")

        def apply(self, current_snapshot, action, *, store, views):
            evidence = store.add_node(
                task_id=current_snapshot.task_id,
                branch_id=current_snapshot.branch_id,
                logical_id="retrieved_second_hop",
                node_type="supporting_fact",
                content={"text": "environment-provided evidence"},
                owner="tool",
                created_by_role="tool",
                provenance={"source": "test_retriever"},
            )
            views.grant(current_snapshot.receiver, node_ids=[evidence.node_id])
            return CommunicationPreview(
                action=action,
                legal=True,
                reason="retrieved from environment adapter",
                closure_node_ids=(evidence.node_id,),
                communication_tokens=3,
            )

    workflow.restore_communication_snapshot(snapshot)
    action = CommunicationAction(
        CommunicationActionType.RETRIEVE,
        query="missing second hop",
    )
    result = CommunicationActionExecutor(information_adapter=Adapter()).apply(
        snapshot,
        action,
        store=workflow.store,
        views=workflow.agent_views,
    )

    assert result.legal
    retrieved = workflow.store.latest_valid("phase2", "main", "retrieved_second_hop")
    assert retrieved is not None
    assert retrieved.node_id in workflow.agent_views.view("solver").visible_node_ids


def test_phase2a_keeps_current_answer_truth_separate_from_final_success():
    snapshot, _, _ = _decision_snapshot()
    optional = tuple(
        item["fragment_id"] for item in snapshot.fragments
        if item["scope"] == "OPTIONAL_POLICY"
    )
    useful = optional[0]

    def runner(action):
        success = action.action_type == CommunicationActionType.SEND and useful in action.selected_fragment_ids
        return _outcome(
            action,
            current=False,
            success=success,
            cost=1 + len(action.selected_fragment_ids),
        )

    result = SendCounterfactualTeacher(exhaustive_limit=10).search(snapshot, runner)

    assert result.status == "teacher_found"
    assert result.teacher_outcomes
    teacher = result.teacher_outcomes[0]
    assert teacher.current_answer.correct is False
    assert teacher.final_outcome.success is True
    assert useful in teacher.action.selected_fragment_ids


def test_phase2a_routes_state_to_phase2b_when_no_send_can_succeed():
    snapshot, _, _ = _decision_snapshot()

    result = SendCounterfactualTeacher().search(
        snapshot,
        lambda action: _outcome(action, current=False, success=False),
    )

    assert result.status == "no_successful_send"
    assert result.teacher_outcomes == ()
    assert "Phase 2B" in result.reason


def test_phase2b_can_select_retrieve_over_stop_send_request_and_revise():
    snapshot, _, _ = _decision_snapshot()
    actions = [
        CommunicationAction(CommunicationActionType.STOP),
        CommunicationAction(CommunicationActionType.SEND),
        CommunicationAction(CommunicationActionType.REQUEST, requested_semantics=("second_hop",)),
        CommunicationAction(CommunicationActionType.RETRIEVE, query="second hop"),
        CommunicationAction(CommunicationActionType.REVISE, target="answer"),
    ]

    def runner(action):
        return _outcome(
            action,
            current=False,
            success=action.action_type == CommunicationActionType.RETRIEVE,
            cost=5,
        )

    result = InformationAcquisitionTeacher().search(snapshot, actions, runner)

    assert result.status == "teacher_found"
    assert result.teacher_outcomes[0].action.action_type == CommunicationActionType.RETRIEVE
    assert result.teacher_outcomes[0].current_answer.correct is False
    assert result.teacher_outcomes[0].final_outcome.success is True


def test_dual_risk_example_keeps_oracle_labels_out_of_features():
    snapshot, _, _ = _decision_snapshot()
    example = build_dual_risk_example(
        snapshot,
        current_answer=CurrentAnswerLabel(correct=False, label_source="oracle"),
        final_outcome=FinalOutcomeLabel(success=True, policy_name="repair-policy"),
    )

    assert "correct" not in example.state_features
    assert "final_task_success" not in example.state_features
    assert example.policy_features == {"policy_name": "repair-policy"}
    assert example.current_answer.correct is False
    assert example.final_outcome.success is True

    try:
        assert_deployable_features({"nested": {"gold_answer": "leak"}})
    except ValueError as exc:
        assert "gold_answer" in str(exc)
    else:
        raise AssertionError("oracle feature was not rejected")


def test_counterfactual_replay_runner_uses_fresh_workflow_per_action():
    snapshot, original_workflow, _ = _decision_snapshot()
    factory_calls = []

    def factory(_snapshot):
        factory_calls.append(1)
        return LangGraphWorkflow(
            store=GraphStore(),
            model=lambda request: {},
            task_id="phase2",
            communication_budget_tokens=256,
        )

    runner = CounterfactualReplayRunner(
        snapshot,
        workflow_factory=factory,
        current_answer_labeler=lambda _snapshot: CurrentAnswerLabel(
            correct=False,
            candidate_node_id="initial-answer",
            label_source="oracle",
        ),
        continuation=lambda workflow, state, action: {
            "final_success": True,
            "final_answer": "repaired",
            "model_input_tokens": 20,
            "model_output_tokens": 4,
        },
    )
    action = CommunicationAction(CommunicationActionType.SEND)

    first = runner(action)
    second = runner(action)

    assert len(factory_calls) == 2
    assert first.current_answer.correct is False
    assert first.final_outcome.success is True
    assert first.final_answer == "repaired"
    assert first.total_cost_tokens == second.total_cost_tokens
    assert original_workflow.store.snapshot().nodes == snapshot.graph_state.nodes


def test_dual_head_network_verify_head_is_policy_independent_and_masks_missing_labels():
    torch = __import__("torch")
    torch.manual_seed(0)
    network = build_dual_head_network(3, 2, hidden_dim=8)
    state = torch.tensor([[1.0, 2.0, 3.0]])
    left_policy = torch.tensor([[1.0, 0.0]])
    right_policy = torch.tensor([[0.0, 1.0]])

    left_current, left_success = network(state, left_policy)
    right_current, right_success = network(state, right_policy)

    assert torch.equal(left_current, right_current)
    assert left_success.shape == right_success.shape == (1,)
    loss = masked_dual_head_bce_loss(
        torch.cat((left_current, right_current)),
        torch.cat((left_success, right_success)),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([float("nan"), 1.0]),
    )
    assert torch.isfinite(loss)


def test_phase2_artifacts_keep_task_splits_disjoint(tmp_path):
    snapshot, _, _ = _decision_snapshot()
    example = build_dual_risk_example(
        snapshot,
        current_answer=CurrentAnswerLabel(correct=False, label_source="oracle"),
        final_outcome=FinalOutcomeLabel(success=True, policy_name="retrieve-policy"),
        policy_features={"action_type": "RETRIEVE"},
    )
    writer = Phase2ArtifactWriter(tmp_path)
    writer.write_risk(example)

    rows = load_jsonl(writer.risk_path)
    assert rows[0]["labels"]["current_answer"]["correct"] is False
    assert rows[0]["labels"]["final_outcome"]["success"] is True
    assert_task_disjoint_splits({"train": rows, "dev": ({"task_id": "another"},)})

    try:
        assert_task_disjoint_splits({"train": rows, "test": rows})
    except ValueError as exc:
        assert "task leakage" in str(exc)
    else:
        raise AssertionError("cross-split task leakage was not rejected")
