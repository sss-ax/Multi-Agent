from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import (
    GraphStore,
    build_role_semantic_contract,
    default_semantic_contract,
    global_semantic_requirements,
)


def test_phase1_all_roles_construct_nonempty_contracts() -> None:
    for role in ("planner", "solver", "critic", "final_solver"):
        contract = build_role_semantic_contract(role, task_type="numeric_solve", domain="gsm8k")

        assert contract.role == role
        assert contract.receiver_role == role
        assert contract.requirements
        for requirement in contract.requirements:
            assert requirement.requirement_id
            assert requirement.semantic_kind


def test_phase1_contract_is_deterministic_for_same_domain_role_task_type() -> None:
    for role in ("planner", "solver", "critic", "final_solver"):
        c1 = build_role_semantic_contract(role, task_type="numeric_solve", domain="gsm8k")
        c2 = build_role_semantic_contract(role, task_type="numeric_solve", domain="gsm8k")
        assert c1 == c2


def test_phase1_contract_does_not_bind_canonical_node_ids() -> None:
    node_id_pattern = re.compile(r"(@v\d+|^node_\d+$|^[a-zA-Z_]+@v\d+$)")

    for role in ("planner", "solver", "critic", "final_solver"):
        contract = build_role_semantic_contract(role, task_type="numeric_solve")
        for requirement in contract.requirements:
            assert requirement.target_node_id is None
            assert not node_id_pattern.search(requirement.requirement_id)
            assert not node_id_pattern.search(requirement.semantic_kind)
            assert not node_id_pattern.search(requirement.target_logical_id or "")
            assert not node_id_pattern.search(requirement.required_type or "")


def test_phase1_contract_does_not_resolve_latest_graph_versions() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="q", owner="user")
    first = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": 1},
        owner="solver",
    )
    second = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"value": 2},
        owner="solver",
    )
    assert first.node_id == "result@v1"
    assert second.node_id == "result@v2"

    contract = build_role_semantic_contract("final_solver", task_type="numeric_solve")

    assert [requirement.semantic_kind for requirement in contract.requirements] == [
        "final_candidate",
        "validation_signal",
    ]
    assert all("result@v" not in requirement.requirement_id for requirement in contract.requirements)
    assert all("result@v" not in (requirement.target_logical_id or "") for requirement in contract.requirements)


def test_phase1_global_and_stage_semantics_are_separated() -> None:
    global_kinds = {requirement.semantic_kind for requirement in global_semantic_requirements(task_type="numeric_solve")}
    stage_kinds = set()
    for sender, receiver in (
        ("planner", "solver"),
        ("solver", "critic"),
        ("critic", "final_solver"),
    ):
        stage_kinds.update(
            requirement.semantic_kind
            for requirement in default_semantic_contract(sender, receiver, task_type="numeric_solve").requirements
        )

    assert global_kinds
    assert stage_kinds
    assert not (global_kinds & stage_kinds)


def test_phase1_gate_metrics_are_clean() -> None:
    contracts = [
        build_role_semantic_contract(role, task_type="numeric_solve")
        for role in ("planner", "solver", "critic", "final_solver")
    ]
    contract_coverage = sum(1 for contract in contracts if contract.requirements) / len(contracts)
    implicit_node_binding = sum(
        1
        for contract in contracts
        for requirement in contract.requirements
        if requirement.target_node_id is not None
    )
    implicit_latest_resolution = sum(
        1
        for contract in contracts
        for requirement in contract.requirements
        if "@v" in requirement.requirement_id or "@v" in (requirement.target_logical_id or "")
    )
    nondeterministic = sum(
        1
        for role in ("planner", "solver", "critic", "final_solver")
        if build_role_semantic_contract(role, task_type="numeric_solve")
        != build_role_semantic_contract(role, task_type="numeric_solve")
    )

    assert contract_coverage == 1.0
    assert implicit_node_binding == 0
    assert implicit_latest_resolution == 0
    assert nondeterministic == 0
