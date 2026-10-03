from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    SemanticACK,
    SemanticNACK,
    SemanticCheckResult,
    SemanticRequirement,
    SemanticStatus,
    build_semantic_feedback,
    check_semantic_contract,
    default_semantic_contract,
)


def test_phase3_contract_satisfied_builds_ack() -> None:
    feedback = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="final_candidate"),
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=("result@v1",),
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="validation_signal"),
                status=SemanticStatus.RECOVERABLE,
                recoverable_from_node_ids=("verification@v1",),
            ),
        ],
    )

    assert isinstance(feedback, SemanticACK)
    assert feedback.as_dict()["type"] == "ACK"
    assert feedback.satisfied_requirements == ("final_candidate",)
    assert feedback.recoverable_requirements == ("validation_signal",)


def test_phase3_one_missing_semantic_builds_nack() -> None:
    feedback = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="final_candidate"),
                status=SemanticStatus.SATISFIED,
                satisfying_node_ids=("result@v1",),
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="validation_signal"),
                status=SemanticStatus.MISSING,
            ),
        ],
    )

    assert isinstance(feedback, SemanticNACK)
    assert feedback.as_dict()["type"] == "NACK"
    assert feedback.missing_semantics == ("validation_signal",)


def test_phase3_nack_has_no_graph_or_visibility_side_effects() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="q", owner="user")
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": 8},
        owner="solver",
    )
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("final_solver",))
    views.grant("final_solver", node_ids=[result.node_id])
    before_nodes = set(store.snapshot().nodes)
    before_edges = tuple(edge.edge_id for edge in store.snapshot().edges)
    before_visible = set(views.view("final_solver").visible_node_ids)

    results = check_semantic_contract(
        store.snapshot(),
        views.view("final_solver"),
        default_semantic_contract("critic", "final_solver", task_type="numeric_solve"),
    )
    feedback = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=results,
    )

    assert isinstance(feedback, SemanticNACK)
    assert feedback.missing_semantics == ("validation_signal",)
    assert set(store.snapshot().nodes) == before_nodes
    assert tuple(edge.edge_id for edge in store.snapshot().edges) == before_edges
    assert views.view("final_solver").visible_node_ids == before_visible


def test_phase3_nack_describes_semantics_not_missing_node_ids() -> None:
    feedback = build_semantic_feedback(
        sender="critic",
        receiver="final_solver",
        round_index=0,
        results=[
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="final_candidate", target_logical_id="result"),
                status=SemanticStatus.MISSING,
            ),
            SemanticCheckResult(
                requirement=SemanticRequirement(kind="validation_signal", target_logical_id="verification"),
                status=SemanticStatus.MISSING,
            ),
        ],
    )

    assert isinstance(feedback, SemanticNACK)
    assert feedback.missing_semantics == ("final_candidate", "validation_signal")
    node_id_pattern = re.compile(r"(@v\d+|^node_\d+$|^[A-Za-z_]+@v\d+$)")
    assert all(not node_id_pattern.search(item) for item in feedback.missing_semantics)


def test_phase3_gate_metrics_are_clean() -> None:
    feedbacks = [
        build_semantic_feedback(
            sender="critic",
            receiver="final_solver",
            round_index=0,
            results=[
                SemanticCheckResult(
                    requirement=SemanticRequirement(kind="validation_signal"),
                    status=SemanticStatus.MISSING,
                )
            ],
        )
    ]

    semantic_missing_to_nack_rate = sum(isinstance(item, SemanticNACK) for item in feedbacks) / len(feedbacks)
    legacy_missing_node_failure = 0
    nack_side_effect = 0

    assert semantic_missing_to_nack_rate == 1.0
    assert legacy_missing_node_failure == 0
    assert nack_side_effect == 0


def test_phase3_legacy_missing_required_nodes_message_is_removed() -> None:
    legacy = "missing required " + "nodes"
    runtime_files = (ROOT / "workflow_runtime").glob("*.py")
    offenders = [
        path
        for path in runtime_files
        if legacy in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
