from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from workflow_runtime import BranchError, GraphStore
from workflow_runtime.langgraph_workflow import LangGraphWorkflow


def make_store() -> GraphStore:
    store = GraphStore()
    task = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="task",
        node_type="task",
        content="solve",
        owner="user",
    )
    query = store.add_node(
        task_id="t",
        branch_id="main",
        logical_id="query_spec",
        node_type="query_spec",
        content={"question": "solve"},
        owner="planner",
    )
    store.add_edge(source=task.node_id, target=query.node_id, relation="depends_on")
    return store


def test_branch_fork_isolated_and_merge_preserves_provenance():
    store = make_store()
    store.create_branch("t", "candidate-a")
    branch_task = store.latest_valid("t", "candidate-a", "task")
    assert branch_task is not None
    assert branch_task.node_id.startswith("candidate-a:")
    branch_result = store.add_node(
        task_id="t",
        branch_id="candidate-a",
        logical_id="result",
        node_type="result",
        content={"value": 7},
        owner="solver",
    )
    assert store.latest_valid("t", "main", "result") is None

    merged = store.merge_branch("t", "candidate-a", logical_ids=["result"])
    assert merged
    result = store.latest_valid("t", "main", "result")
    assert result is not None
    assert result.content == {"value": 7}
    assert result.provenance["merged_from_branch"] == "candidate-a"
    assert result.provenance["merged_from_node"] == branch_result.node_id


def test_branch_workflow_uses_distinct_session_identity():
    store = make_store()
    workflow = LangGraphWorkflow(store=store, model=lambda request: {}, task_id="t")
    branch = workflow.fork("candidate-b")
    assert branch.branch_id == "candidate-b"
    assert branch.session_id == "workflow:t:candidate-b"
    assert store.branch_digest("t", "main") != store.branch_digest("t", "candidate-b")


def test_branch_errors_are_explicit():
    store = make_store()
    with pytest.raises(BranchError):
        store.create_branch("t", "missing-source", source_branch_id="unknown")
    store.create_branch("t", "candidate")
    with pytest.raises(BranchError):
        store.create_branch("t", "candidate")
