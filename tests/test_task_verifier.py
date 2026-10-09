from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow_runtime import GraphStore
from workflow_runtime.domain_executors import execute_python_tests_detailed
from workflow_runtime.task_verifier import apply_verification_signal, verify_task_candidate


def test_code_verifier_forces_need_fix_when_tests_fail() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="code", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="execution",
        node_type="execution",
        content={"status": "failed", "stderr": "AssertionError"},
        owner="tool",
        status="need_fix",
        validation={"schema_valid": True, "execution_valid": False},
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"tests_passed": False, "errors": ["AssertionError"]},
        owner="tool",
        validation={"schema_valid": True, "execution_valid": False},
    )

    signal = verify_task_candidate("code_generation", store, task_id="t", branch_id="main")
    action = apply_verification_signal(
        {"op": "verify", "target": "result", "status": "verified"},
        signal,
    )

    assert signal.status == "need_fix"
    assert signal.error_type == "execution_failure"
    assert action["status"] == "need_fix"
    assert action["repair_instruction"]


def test_mbpp_executor_extracts_assertion_expected_actual() -> None:
    result = execute_python_tests_detailed(
        code="def add_one(x):\n    return x",
        tests=["assert add_one(1) == 2"],
    )

    assert not result.success
    assert result.execution["kind"] == "assertion_failure"
    assert result.execution["function_name"] == "add_one"
    assert result.execution["expected"] == "2"
    assert result.execution["actual"] == "1"


def test_code_verifier_uses_structured_assertion_failure() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="code", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="execution",
        node_type="execution",
        content={
            "status": "failed",
            "kind": "assertion_failure",
            "function_name": "add_one",
            "failed_test": "assert add_one(1) == 2",
            "expected": "2",
            "actual": "1",
        },
        owner="tool",
        status="need_fix",
        validation={"schema_valid": True, "execution_valid": False},
    )

    signal = verify_task_candidate("code_generation", store, task_id="t", branch_id="main")

    assert signal.status == "need_fix"
    assert signal.error_type == "wrong_value"
    assert "assert add_one(1) == 2" in signal.error_location
    assert "returned 1" in signal.repair_instruction
    assert "must return 2" in signal.repair_instruction


def test_choice_verifier_reports_invalid_option() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="mcq", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="choice_1",
        node_type="choice",
        content={"label": "A", "text": "alpha"},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "answer", "value": "Z"},
        owner="solver",
    )

    signal = verify_task_candidate("multiple_choice", store, task_id="t", branch_id="main")

    assert signal.status == "need_fix"
    assert signal.error_type == "invalid_option"


def test_math_verifier_accepts_repaired_result_with_stale_calculation_trace() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="2+3", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="calculation",
        node_type="calculation",
        content={"id": "R1", "expression": "2+3", "value": 6},
        owner="solver",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "R1", "value": 5},
        owner="solver",
    )

    signal = verify_task_candidate("numeric_solve", store, task_id="t", branch_id="main")

    assert signal.status == "verified"
    assert "calculation_trace_stale" in signal.deterministic_checks


def test_choice_verifier_rejects_bare_choice_label() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="mcq", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="choice_schema",
        node_type="choice_schema",
        content='{"allowed_labels":["A","B","C","D","E","F","G","H","I","J"]}',
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="choice_9",
        node_type="choice",
        content='{"label":"I","text":"iota"}',
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content='{"id":"answer","value":"I"}',
        owner="solver",
    )

    signal = verify_task_candidate("multiple_choice", store, task_id="t", branch_id="main")

    assert signal.status == "need_fix"
    assert signal.error_type == "missing_semantic_option_verification"


def test_choice_verifier_uses_structured_independent_option_analysis() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="mcq", owner="user")
    for label in ("A", "B"):
        store.add_node(
            task_id="t",
            branch_id="main",
            logical_id=f"choice_{label.lower()}",
            node_type="choice",
            content={"label": label, "text": f"option {label}"},
            owner="dataset",
        )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={
            "id": "answer",
            "value": {
                "answer": "B",
                "solver_choice": "B",
                "independent_choice": "B",
                "option_analysis": {
                    "A": {"support": [], "contradiction": ["not supported"]},
                    "B": {"support": ["supported"], "contradiction": []},
                },
                "confidence": 0.8,
            },
        },
        owner="solver",
    )

    signal = verify_task_candidate("multiple_choice", store, task_id="t", branch_id="main")

    assert signal.status == "verified"
    assert "independent_choice_agrees" in signal.consistency_checks
    assert signal.metadata["independent_choice"] == "B"


def test_choice_verifier_rejects_structured_disagreement() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="mcq", owner="user")
    for label in ("A", "B"):
        store.add_node(
            task_id="t",
            branch_id="main",
            logical_id=f"choice_{label.lower()}",
            node_type="choice",
            content={"label": label, "text": f"option {label}"},
            owner="dataset",
        )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={
            "id": "answer",
            "value": {
                "answer": "B",
                "solver_choice": "B",
                "independent_choice": "A",
                "option_analysis": {
                    "A": {"support": ["supported"], "contradiction": []},
                    "B": {"support": [], "contradiction": ["not supported"]},
                },
                "confidence": 0.8,
            },
        },
        owner="solver",
    )

    signal = verify_task_candidate("multiple_choice", store, task_id="t", branch_id="main")

    assert signal.status == "need_fix"
    assert signal.error_type == "semantic_disagreement"


def test_hotpot_verifier_requires_structured_citations() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="qa", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_1",
        node_type="supporting_fact",
        content={"title": "Ada", "sent_id": 0, "text": "Ada was born in London."},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={"id": "answer", "value": "London"},
        owner="solver",
    )

    signal = verify_task_candidate("multihop_qa", store, task_id="t", branch_id="main")

    assert signal.status == "need_fix"
    assert signal.error_type == "missing_evidence_chain"


def test_hotpot_verifier_accepts_structured_bridge_chain() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="qa", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_1",
        node_type="supporting_fact",
        content={"title": "Ada", "sent_id": 0, "text": "Ada collaborated with Charles Babbage."},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_2",
        node_type="supporting_fact",
        content={"title": "Charles Babbage", "sent_id": 0, "text": "Charles Babbage was born in London."},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={
            "id": "answer",
            "value": {
                "answer": "London",
                "bridge_entity": "Charles Babbage",
                "cited_fact_ids": ["supporting_fact_1", "supporting_fact_2"],
                "reasoning_chain": [
                    "supporting_fact_1 -> Charles Babbage",
                    "Charles Babbage + supporting_fact_2 -> London",
                ],
            },
        },
        owner="solver",
    )

    signal = verify_task_candidate("multihop_qa", store, task_id="t", branch_id="main")

    assert signal.status == "verified"
    assert "chain_complete" in signal.consistency_checks
    assert "semantic_entailment_delegated_to_critic" in signal.consistency_checks
    assert signal.confidence == 0.0
    assert signal.metadata["answer"] == "London"
    assert signal.metadata["semantic_entailment_required_from"] == "critic"


def test_hotpot_verifier_does_not_use_lexical_answer_proxy() -> None:
    store = GraphStore()
    store.add_node(task_id="t", branch_id="main", logical_id="task", node_type="task", content="qa", owner="user")
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_1",
        node_type="supporting_fact",
        content={"title": "Ada", "sent_id": 0, "text": "Ada collaborated with Charles Babbage."},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="supporting_fact_2",
        node_type="supporting_fact",
        content={"title": "Charles Babbage", "sent_id": 0, "text": "Charles Babbage designed the Analytical Engine."},
        owner="dataset",
    )
    store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="result",
        node_type="result",
        content={
            "id": "answer",
            "value": {
                "answer": "Paris",
                "bridge_entity": "Charles Babbage",
                "cited_fact_ids": ["supporting_fact_1", "supporting_fact_2"],
                "reasoning_chain": [
                    "supporting_fact_1 -> Charles Babbage",
                    "Charles Babbage + supporting_fact_2 -> Paris",
                ],
            },
        },
        owner="solver",
    )

    signal = verify_task_candidate("multihop_qa", store, task_id="t", branch_id="main")

    assert signal.status == "verified"
    assert "semantic_entailment_delegated_to_critic" in signal.consistency_checks
    assert signal.confidence == 0.0
