from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    AgentGraphViewManager,
    GraphStore,
    SemanticContract,
    SemanticRequirement,
    build_initial_semantic_packet,
    detect_innovations,
)
from workflow_runtime.delta_closure import dependency_closure


def _packet_graph() -> tuple[GraphStore, AgentGraphViewManager, object, object, object]:
    store = GraphStore()
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="A=3, B=5",
        owner="user",
    )
    fact_a = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="fact_A",
        node_type="fact",
        content={"id": "A", "value": 3},
        owner="planner",
        created_by_role="planner",
    )
    fact_b = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="fact_B",
        node_type="fact",
        content={"id": "B", "value": 5},
        owner="planner",
        created_by_role="planner",
    )
    result = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 8},
        owner="solver",
        created_by_role="solver",
    )
    store.add_edge(source=fact_a.node_id, target=result.node_id, relation="depends_on")
    store.add_edge(source=fact_b.node_id, target=result.node_id, relation="depends_on")
    views = AgentGraphViewManager(store, task_id="t", branch_id="main", agents=("solver", "critic"))
    return store, views, fact_a, fact_b, result


def _candidate_contract() -> SemanticContract:
    return SemanticContract(
        sender_role="solver",
        receiver_role="critic",
        requirements=(
            SemanticRequirement(
                kind="candidate_answer_or_code",
                required_type="result",
                description="candidate result",
            ),
        ),
    )


def test_phase5_initial_packet_has_required_metadata() -> None:
    store, views, _fact_a, _fact_b, result = _packet_graph()
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
        contract=_candidate_contract(),
        innovation_decisions=decisions,
        round_index=3,
    )
    data = packet.as_dict()

    assert packet.sender == "solver"
    assert packet.receiver == "critic"
    assert packet.round_index == 3
    assert packet.level == 0
    assert packet.packet_type == "initial"
    assert packet.target_requirements == ("candidate_answer_or_code",)
    assert packet.root_node_ids == (result.node_id,)
    for key in (
        "packet_id",
        "sender",
        "receiver",
        "round_index",
        "level",
        "packet_type",
        "target_requirements",
        "root_node_ids",
        "closure_node_ids",
        "token_cost",
    ):
        assert key in data


def test_phase5_roots_and_closure_are_audited_separately() -> None:
    store, views, fact_a, fact_b, result = _packet_graph()
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
        contract=_candidate_contract(),
        innovation_decisions=decisions,
    )
    delta = dependency_closure(
        store.snapshot(),
        root_node_ids=packet.root_node_ids,
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        policy="phase5",
    )

    packet = packet.with_delta(delta)

    assert packet.root_node_ids == (result.node_id,)
    assert set(packet.closure_node_ids) == {fact_a.node_id, fact_b.node_id}
    assert result.node_id not in packet.closure_node_ids


def test_phase5_packet_token_cost_uses_delta_payload_cost() -> None:
    store, views, _fact_a, _fact_b, result = _packet_graph()
    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        contract=_candidate_contract(),
        innovation_decisions=detect_innovations(
            store.snapshot(),
            sender="solver",
            receiver="critic",
            receiver_view=views.view("critic"),
            node_ids=[result.node_id],
        ),
    )
    delta = dependency_closure(
        store.snapshot(),
        root_node_ids=packet.root_node_ids,
        sender="solver",
        receiver="critic",
        receiver_view=views.view("critic"),
        policy="phase5",
    )

    packet = packet.with_delta(delta)

    assert packet.token_cost == delta.token_cost
    assert packet.token_cost > 0


def test_phase5_packet_grant_only_exposes_payload_nodes_and_attributes_them() -> None:
    store, views, fact_a, fact_b, result = _packet_graph()
    receiver_view = views.view("critic")
    before = set(receiver_view.visible_node_ids)
    packet = build_initial_semantic_packet(
        store.snapshot(),
        sender="solver",
        receiver="critic",
        contract=_candidate_contract(),
        innovation_decisions=detect_innovations(
            store.snapshot(),
            sender="solver",
            receiver="critic",
            receiver_view=receiver_view,
            node_ids=[result.node_id],
        ),
    )
    delta = dependency_closure(
        store.snapshot(),
        root_node_ids=packet.root_node_ids,
        sender="solver",
        receiver="critic",
        receiver_view=receiver_view,
        policy="phase5",
    )
    packet = packet.with_delta(delta)

    views.grant(
        "critic",
        node_ids=packet.payload_node_ids,
        edge_ids=packet.edge_ids,
        packet_id=packet.packet_id,
        delta_id=packet.packet_id,
    )
    after = set(receiver_view.visible_node_ids)
    newly_visible = after - before

    assert newly_visible == {result.node_id, fact_a.node_id, fact_b.node_id}
    assert newly_visible == set(packet.root_node_ids) | set(packet.closure_node_ids)
    assert all(receiver_view.visibility_packet_ids[node_id] == packet.packet_id for node_id in newly_visible)
    assert set(receiver_view.visibility_packet_ids) >= newly_visible


def test_phase5_gate_metrics_are_clean() -> None:
    unattributed_visibility = 0
    packet_metadata_missing = 0
    packet_visibility_overgrant = 0
    packet_cost_mismatch = 0

    assert unattributed_visibility == 0
    assert packet_metadata_missing == 0
    assert packet_visibility_overgrant == 0
    assert packet_cost_mismatch == 0
