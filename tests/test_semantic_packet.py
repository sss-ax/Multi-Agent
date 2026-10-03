from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    DECISION_SUFFICIENT,
    GraphStore,
    SemanticContract,
    SemanticPacketBuilder,
    build_initial_semantic_packet,
    default_semantic_contract,
    detect_innovations,
)


def _store_and_views() -> tuple[GraphStore, AgentGraphViewManager]:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    return store, views


def test_initial_packet_selects_only_innovative_contract_relevant_roots() -> None:
    store, views = _store_and_views()
    calculation = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3", "value": 5},
        owner="solver",
    )
    views.grant("critic", node_ids=[calculation.node_id])
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
    )
    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[result.node_id, verification.node_id],
    )
    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")

    packet = SemanticPacketBuilder(store.snapshot()).build_initial(
        sender="solver",
        receiver="critic",
        contract=contract,
        innovation_decisions=decisions,
    )

    assert packet.level == DECISION_SUFFICIENT
    assert packet.is_initial
    assert not packet.is_fallback
    assert packet.root_node_ids == ()
    assert packet.closure_node_ids == ()
    assert packet.edge_ids == ()
    assert packet.token_cost == 0


def test_initial_packet_orders_relevant_innovations_by_semantic_priority() -> None:
    store, views = _store_and_views()
    calculation = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3"},
        owner="solver",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[calculation.node_id, result.node_id],
    )
    contract = default_semantic_contract("solver", "critic", task_type="numeric_solve")

    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        contract=contract,
        innovation_decisions=decisions,
    )

    assert packet.root_node_ids == (result.node_id,)
    assert packet.target_requirements == ("candidate_answer_or_code",)
    assert packet.token_cost > 0
    assert packet.packet_id.startswith("packet:solver->critic:r0:l0:")


def test_initial_packet_ignores_non_contract_innovations_when_contract_is_explicit() -> None:
    store, views = _store_and_views()
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
    )
    decisions = detect_innovations(
        store.snapshot(),
        sender="critic",
        receiver="solver",
        receiver_view=views.view("solver"),
        node_ids=[verification.node_id],
    )
    contract = default_semantic_contract("planner", "solver", task_type="numeric_solve")

    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="critic",
        receiver="solver",
        contract=contract,
        innovation_decisions=decisions,
    )

    assert packet.root_node_ids == ()
    assert packet.target_requirements == ()


def test_packet_as_dict_is_stable() -> None:
    store, views = _store_and_views()
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )
    decisions = detect_innovations(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        node_ids=[result.node_id],
    )
    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        contract=default_semantic_contract("solver", "critic", task_type="numeric_solve"),
        innovation_decisions=decisions,
        round_index=2,
    )

    data = packet.as_dict()
    assert data["sender"] == "solver"
    assert data["receiver"] == "critic"
    assert data["round_index"] == 2
    assert data["root_node_ids"] == [result.node_id]
    assert data["is_initial"] is True
    assert data["is_fallback"] is False


def test_initial_packet_with_empty_contract_sends_no_unsolicited_root() -> None:
    store, views = _store_and_views()
    verification = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="verification",
        node_type="verification",
        content={"target": "result", "status": "verified"},
        owner="critic",
    )
    decisions = detect_innovations(
        store.snapshot(),
        sender="critic",
        receiver="solver",
        receiver_view=views.view("solver"),
        node_ids=[verification.node_id],
    )

    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="critic",
        receiver="solver",
        contract=SemanticContract(sender_role="critic", receiver_role="solver"),
        innovation_decisions=decisions,
    )

    assert packet.root_node_ids == ()
    assert packet.target_requirements == ()
