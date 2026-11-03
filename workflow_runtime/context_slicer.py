"""Dependency-aware context slicing and deterministic materialization."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .canonical import canonical_json, dependency_digest, node_order_digest
from .graph_store import GraphStore, GraphValidationError
from .models import ContextSlice, DependencyEdge, GraphState, NodeVersion
from .relations import DATA_DEPENDENCY_RELATIONS, VALIDATION_RELATIONS
from .semantic_fragments import fragments_for_node, render_fragmented_content
from .task_spec import get_task_spec
from .hotpot_graph_retriever import GraphSeedRetriever


class SliceError(ValueError):
    """Raised when a role cannot receive a sound context slice."""


ROLE_REQUIREMENTS: Dict[str, List[str]] = {
    "planner": ["task"],
    # Solver roots are selected by task_type below.  This list is the default
    # contract for numeric workflows and remains public for callers that
    # inspect the runtime requirements.
    "solver": ["query_spec", "facts", "plan", "plan_steps"],
    "critic": ["task", "query_spec", "facts", "calculation", "result"],
    "final_solver": ["query_spec", "result", "verification"],
}

NODE_ORDER = {
    "task": 0,
    "source": 1,
    "query_spec": 2,
    "facts": 3,
    "fact": 4,
    "evidence": 4,
    "choice": 4,
    "table": 4,
    "table_cell": 5,
    "entity": 4,
    "supporting_fact": 5,
    "evidence_link": 6,
    "requirements": 4,
    "code": 7,
    "test": 5,
    "execution": 8,
    "error": 9,
    "tool_request": 8,
    "tool_result": 9,
    "plan": 10,
    "plan_steps": 11,
    "calculation": 12,
    "result": 13,
    "verification": 14,
    "final_answer": 15,
    "task_source": 1,
    "question_source": 1,
    "document_source": 1,
    "sentence_source": 1,
    "entity_source": 1,
    "choice_source": 4,
    "requirements_source": 4,
    "entry_point_source": 4,
    "public_test_source": 5,
    "plan_step": 11,
    "candidate_answer": 13,
    "slot_analysis": 12,
    "selected_evidence": 6,
    "bridge_hypothesis": 7,
    "reasoning_step": 12,
    "semantic_delta_boundary": 16,
}

DOMAIN_SOLVER_REQUIREMENTS: Dict[str, List[str]] = {
    "table_qa": ["query_spec", "table", "table_cell", "evidence", "plan", "plan_steps"],
    "multihop_qa": ["query_spec", "entity", "supporting_fact", "evidence_link", "plan", "plan_steps"],
    "multiple_choice": ["query_spec", "choice", "plan", "plan_steps"],
    "code_generation": ["query_spec", "requirements", "plan", "plan_steps"],
    "marble_research": ["query_spec", "plan", "plan_steps"],
    "marble_bargaining": ["query_spec", "plan", "plan_steps"],
    "marble_database": ["query_spec", "requirements", "plan", "plan_steps"],
}

DOMAIN_PLANNER_REQUIREMENTS: Dict[str, List[str]] = {
    "table_qa": ["task", "table", "table_cell", "evidence"],
    "multihop_qa": ["task", "entity", "supporting_fact", "evidence_link"],
    "multiple_choice": ["task", "choice"],
    "code_generation": ["task", "requirements"],
    "marble_research": ["task"],
    "marble_bargaining": ["task"],
    "marble_database": ["task", "requirements"],
}

DOMAIN_CRITIC_REQUIREMENTS: Dict[str, List[str]] = {
    "code_generation": ["task", "query_spec", "code", "execution"],
    "multihop_qa": ["task", "query_spec", "entity", "supporting_fact", "evidence_link", "result"],
    "multiple_choice": ["task", "query_spec", "choice", "result"],
    "marble_research": ["task", "query_spec", "result"],
    "marble_bargaining": ["task", "query_spec", "result"],
    "marble_database": ["task", "query_spec", "result"],
}

DOMAIN_FINAL_REQUIREMENTS: Dict[str, List[str]] = {
    "code_generation": ["query_spec", "code", "verification"],
}

SEMANTIC_BOUNDARY_VALIDATIONS = frozenset({"evidence_checked", "model_judged_correct"})


def _state(store_or_state: GraphStore | GraphState) -> GraphState:
    return store_or_state.snapshot() if isinstance(store_or_state, GraphStore) else store_or_state


def _task_type_from_marker(store: GraphStore, task_id: str, branch_id: str) -> str:
    task = store.latest_valid(task_id, branch_id, "task")
    text = str(task.content).lower() if task is not None else ""
    for marker, task_type in (
        ("[domain=table_qa]", "table_qa"),
        ("[domain=multihop_qa]", "multihop_qa"),
        ("[domain=multiple_choice]", "multiple_choice"),
        ("[domain=code_generation]", "code_generation"),
        ("[domain=marble_research]", "marble_research"),
        ("[domain=marble_bargaining]", "marble_bargaining"),
        ("[domain=marble_database]", "marble_database"),
    ):
        if marker in text:
            return task_type
    return ""


def _task_type_from_query_or_marker(store: GraphStore, task_id: str, branch_id: str) -> str:
    """Resolve the task family from source and runtime node types."""
    query = store.latest_valid(task_id, branch_id, "query_spec")
    if query is not None:
        try:
            value = json.loads(query.content) if isinstance(query.content, str) else query.content
            if isinstance(value, dict) and value.get("task_type"):
                return str(value["task_type"])
        except (TypeError, json.JSONDecodeError):
            pass
    return _task_type_from_marker(store, task_id, branch_id)


def _is_semantic_source_graph(store: GraphStore, task_id: str, branch_id: str) -> bool:
    task = store.latest_valid(task_id, branch_id, "task")
    return task is not None and task.type.endswith("_source")


def _node_sort_key(node: NodeVersion) -> Tuple[int, str, int, str]:
    return NODE_ORDER.get(node.type, 100), node.logical_id, node.version, node.node_id


def upstream_closure(
    store_or_state: GraphStore | GraphState,
    roots: Iterable[str],
    *,
    stop_at_verified: bool = False,
    include_validation_edges: bool = False,
) -> Set[str]:
    state = _state(store_or_state)
    visible: Set[str] = set()
    stack = list(dict.fromkeys(roots))
    incoming: Dict[str, List[DependencyEdge]] = {}
    for edge in state.edges:
        if edge.relation in DATA_DEPENDENCY_RELATIONS:
            incoming.setdefault(edge.target, []).append(edge)
        elif include_validation_edges and edge.relation in VALIDATION_RELATIONS:
            incoming.setdefault(edge.target, []).append(edge)

    while stack:
        node_id = stack.pop()
        if node_id in visible:
            continue
        node = state.nodes.get(node_id)
        if node is None:
            raise SliceError(f"unknown node in dependency closure: {node_id}")
        visible.add(node_id)
        if stop_at_verified and node.is_semantically_verified():
            continue
        for edge in incoming.get(node_id, []):
            # Validation edges are not upstream data.  They are included only
            # when explicitly requested and are traversed to keep provenance.
            stack.append(edge.source)
    return visible


def _required_roots(
    store: GraphStore,
    task_id: str,
    branch_id: str,
    role: str,
) -> Tuple[List[str], List[str]]:
    if role not in ROLE_REQUIREMENTS:
        raise SliceError(f"unknown role: {role}")
    # Semantic Delta v2 starts from typed Source nodes.  TaskSpec context types
    # are selectors, not an assertion that every optional source type exists.
    # In particular, HotpotQA has sentence_source nodes rather than generic
    # supporting_fact/evidence_link pair.  Apply this from the very first
    # planner call, before a semantic boundary exists.
    if _is_semantic_source_graph(store, task_id, branch_id):
        task_type = _task_type_from_query_or_marker(store, task_id, branch_id)
        try:
            allowed_context_types = set(get_task_spec(task_type).context_for(role))
        except ValueError:
            allowed_context_types = set()
        roots = [
            node.node_id
            for node in sorted(store.snapshot().nodes.values(), key=_node_sort_key)
            if node.task_id == task_id
            and node.branch_id == branch_id
            and node.is_operationally_valid()
            and (not allowed_context_types or node.type in allowed_context_types)
        ]
        query = store.latest_valid(task_id, branch_id, "query_spec")
        if query is not None and query.node_id not in roots:
            roots.insert(0, query.node_id)
        task = store.latest_valid(task_id, branch_id, "task")
        if task is not None and role == "planner" and task.node_id not in roots:
            roots.insert(0, task.node_id)
        if not roots:
            return [], [f"TaskSpec context for {task_type or 'unknown'}"]
        return roots, []
    required = ROLE_REQUIREMENTS[role]
    if role == "planner":
        task = store.latest_valid(task_id, branch_id, "task")
        task_text = str(task.content).lower() if task is not None else ""
        for marker, task_type in (
            ("[domain=table_qa]", "table_qa"),
            ("[domain=multihop_qa]", "multihop_qa"),
            ("[domain=multiple_choice]", "multiple_choice"),
            ("[domain=code_generation]", "code_generation"),
            ("[domain=marble_research]", "marble_research"),
            ("[domain=marble_bargaining]", "marble_bargaining"),
            ("[domain=marble_database]", "marble_database"),
        ):
            if marker in task_text:
                required = DOMAIN_PLANNER_REQUIREMENTS[task_type]
                break
    if role == "solver":
        query = store.latest_valid(task_id, branch_id, "query_spec")
        task_type = ""
        if query is not None:
            try:
                value = json.loads(query.content) if isinstance(query.content, str) else query.content
                task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
            except (TypeError, json.JSONDecodeError):
                pass
        task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
        required = DOMAIN_SOLVER_REQUIREMENTS.get(task_type, required)
    if role == "critic":
        query = store.latest_valid(task_id, branch_id, "query_spec")
        task_type = ""
        if query is not None:
            try:
                value = json.loads(query.content) if isinstance(query.content, str) else query.content
                task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
            except (TypeError, json.JSONDecodeError):
                pass
        task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
        required = DOMAIN_CRITIC_REQUIREMENTS.get(task_type, required)
    if role == "final_solver":
        query = store.latest_valid(task_id, branch_id, "query_spec")
        task_type = ""
        if query is not None:
            try:
                value = json.loads(query.content) if isinstance(query.content, str) else query.content
                task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
            except (TypeError, json.JSONDecodeError):
                pass
        task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
        required = DOMAIN_FINAL_REQUIREMENTS.get(task_type, required)
    semantic_nodes = [
        node for node in store.snapshot().nodes.values()
        if node.task_id == task_id
        and node.branch_id == branch_id
        and node.is_operationally_valid()
        and bool(node.provenance.get("semantic_delta"))
    ]
    semantic_boundaries = [
        node for node in store.snapshot().nodes.values()
        if node.task_id == task_id
        and node.branch_id == branch_id
        and node.is_operationally_valid()
        and bool(node.provenance.get("semantic_delta_boundary"))
    ]
    if (semantic_nodes or semantic_boundaries) and role in {"solver", "critic"}:
        query = store.latest_valid(task_id, branch_id, "query_spec")
        task_type = ""
        if query is not None and isinstance(query.content, dict):
            task_type = str(query.content.get("task_type", ""))
        task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
        try:
            allowed_context_types = set(get_task_spec(task_type).context_for(role))
        except ValueError:
            allowed_context_types = set()
        roots = []
        if query is not None:
            roots.append(query.node_id)
        if semantic_nodes:
            roots.extend(
                node.node_id
                for node in sorted(semantic_nodes, key=_node_sort_key)
                if not allowed_context_types or node.type in allowed_context_types
            )
            if role == "critic":
                for logical_id in ("result", "execution", "code"):
                    current = store.latest_valid(task_id, branch_id, logical_id)
                    if current is not None:
                        roots.append(current.node_id)
        else:
            roots.extend(
                node.node_id
                for node in sorted(store.snapshot().nodes.values(), key=_node_sort_key)
                if node.task_id == task_id
                and node.branch_id == branch_id
                and node.is_operationally_valid()
            )
        return roots, []

    roots: List[str] = []
    missing: List[str] = []
    for logical_id in required:
        if logical_id in {"table_cell", "evidence", "entity", "supporting_fact", "evidence_link", "choice", "test"}:
            state = store.snapshot()
            matches = sorted(
                (node for node in state.nodes.values()
                 if node.task_id == task_id and node.branch_id == branch_id
                 and node.logical_id.startswith(logical_id + "_")
                 and node.is_operationally_valid()),
                key=lambda node: node.logical_id,
            )
            if matches:
                roots.extend(node.node_id for node in matches)
            else:
                missing.append(logical_id)
            continue
        node = store.latest_valid(task_id, branch_id, logical_id)
        if node is None:
            missing.append(logical_id)
        else:
            roots.append(node.node_id)
    return roots, missing


def _required_logical_ids(store: GraphStore, task_id: str, branch_id: str, role: str) -> List[str]:
    if role == "planner":
        task = store.latest_valid(task_id, branch_id, "task")
        task_text = str(task.content).lower() if task is not None else ""
        for marker, task_type in (
            ("[domain=table_qa]", "table_qa"),
            ("[domain=multihop_qa]", "multihop_qa"),
            ("[domain=multiple_choice]", "multiple_choice"),
            ("[domain=code_generation]", "code_generation"),
            ("[domain=marble_research]", "marble_research"),
            ("[domain=marble_bargaining]", "marble_bargaining"),
            ("[domain=marble_database]", "marble_database"),
        ):
            if marker in task_text:
                return DOMAIN_PLANNER_REQUIREMENTS[task_type]
        return ROLE_REQUIREMENTS[role]
    if role != "solver":
        if role == "critic":
            query = store.latest_valid(task_id, branch_id, "query_spec")
            task_type = ""
            if query is not None:
                try:
                    value = json.loads(query.content) if isinstance(query.content, str) else query.content
                    task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
                except (TypeError, json.JSONDecodeError):
                    pass
            task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
            return DOMAIN_CRITIC_REQUIREMENTS.get(task_type, ROLE_REQUIREMENTS[role])
        if role == "final_solver":
            query = store.latest_valid(task_id, branch_id, "query_spec")
            task_type = ""
            if query is not None:
                try:
                    value = json.loads(query.content) if isinstance(query.content, str) else query.content
                    task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
                except (TypeError, json.JSONDecodeError):
                    pass
            task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
            return DOMAIN_FINAL_REQUIREMENTS.get(task_type, ROLE_REQUIREMENTS[role])
        return ROLE_REQUIREMENTS[role]
    query = store.latest_valid(task_id, branch_id, "query_spec")
    task_type = ""
    if query is not None:
        try:
            value = json.loads(query.content) if isinstance(query.content, str) else query.content
            task_type = str(value.get("task_type", "")) if isinstance(value, dict) else ""
        except (TypeError, json.JSONDecodeError):
            pass
    task_type = task_type or _task_type_from_marker(store, task_id, branch_id)
    return DOMAIN_SOLVER_REQUIREMENTS.get(task_type, ROLE_REQUIREMENTS[role])


def _token_count(text: str) -> int:
    # This is deliberately a tokenizer-independent lower-level estimate.  The
    # model backend replaces it with the actual tokenizer count at invocation.
    return len(text.split()) if text else 0


def _decoded_content(node: NodeVersion) -> Any:
    content = node.content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _multihop_source_subset(
    store: GraphStore,
    visible_ids: Set[str],
    *,
    task_id: str,
    branch_id: str,
    sentence_limit: int,
) -> Set[str]:
    """Bound large Hotpot source payloads without consulting gold evidence.

    Selection is deterministic and question-guided.  Sentences already used
    as grounding for a derived node are always retained, so later critic and
    repair calls cannot lose committed provenance.  The lexical seed set is
    expanded by document-title mentions to recover the common two-hop bridge
    pattern (question -> first document -> second document).
    """
    if sentence_limit <= 0:
        return visible_ids
    state = store.snapshot()
    sentence_nodes = [
        state.nodes[node_id]
        for node_id in visible_ids
        if node_id in state.nodes and state.nodes[node_id].type == "sentence_source"
    ]
    task = store.latest_valid(task_id, branch_id, "task")
    question = str(task.content if task is not None else "")
    question = re.sub(r"^\[domain=[^\]]+\]\s*", "", question).strip()
    question_folded = question.casefold()
    facts: list[dict[str, Any]] = []
    node_by_ref: dict[str, NodeVersion] = {}
    for node in sentence_nodes:
        value = _decoded_content(node)
        record = dict(value) if isinstance(value, dict) else {"text": str(value)}
        record["fact_id"] = node.logical_id
        facts.append(record)
        node_by_ref[node.logical_id] = node

    ranked = GraphSeedRetriever().select(question, facts, limit=len(facts))
    ranked_refs = [seed.fact_id for seed in ranked if seed.fact_id in node_by_ref]
    # Start from a small high-precision seed set so title expansion gets room
    # inside the final budget instead of being appended after a full top-k.
    selected_refs: list[str] = ranked_refs[: min(8, sentence_limit)]

    # Preserve sources already committed as grounding, independently of their
    # lexical score.  Dataset task->source edges do not match this condition.
    derived_types = {
        node.node_id: node.type
        for node in state.nodes.values()
        if node.task_id == task_id and node.branch_id == branch_id
    }
    for edge in state.edges:
        source = state.nodes.get(edge.source)
        if (
            source is not None
            and source.type == "sentence_source"
            and source.node_id in visible_ids
            and not derived_types.get(edge.target, "").endswith("_source")
            and derived_types.get(edge.target, "") not in {"task", "query_spec"}
        ):
            selected_refs.insert(0, source.logical_id)

    # Expand a seed through titles explicitly mentioned in its text.  This is
    # a deterministic retrieval operation over public context, not gold-label
    # selection.
    title_to_refs: dict[str, list[str]] = {}
    for fact in facts:
        title = str(fact.get("title", "")).strip()
        if title:
            title_to_refs.setdefault(title.casefold(), []).append(str(fact["fact_id"]))
    selected_text = " ".join(
        str(_decoded_content(node_by_ref[ref]).get("text", ""))
        for ref in selected_refs
        if ref in node_by_ref and isinstance(_decoded_content(node_by_ref[ref]), dict)
    ).casefold()
    for title, refs in sorted(title_to_refs.items()):
        if title and (title in question_folded or title in selected_text):
            selected_refs.extend(refs)

    selected_refs.extend(ranked_refs)
    selected_refs = list(dict.fromkeys(selected_refs))[:sentence_limit]
    selected_sentence_ids = {node_by_ref[ref].node_id for ref in selected_refs if ref in node_by_ref}
    selected_titles = {
        str((_decoded_content(state.nodes[node_id]) or {}).get("title", "")).casefold()
        for node_id in selected_sentence_ids
        if isinstance(_decoded_content(state.nodes[node_id]), dict)
    }
    selected_text = " ".join(
        str((_decoded_content(state.nodes[node_id]) or {}).get("text", ""))
        for node_id in selected_sentence_ids
        if isinstance(_decoded_content(state.nodes[node_id]), dict)
    ).casefold()
    result: Set[str] = set()
    entity_count = 0
    for node_id in visible_ids:
        node = state.nodes.get(node_id)
        if node is None:
            continue
        if node.type == "sentence_source":
            if node_id in selected_sentence_ids:
                result.add(node_id)
            continue
        if node.type == "document_source":
            value = _decoded_content(node)
            title = str(value.get("title", "") if isinstance(value, dict) else value).casefold()
            if title in selected_titles or (title and title in question_folded):
                result.add(node_id)
            continue
        if node.type == "entity_source":
            value = _decoded_content(node)
            name = str(value.get("name", "") if isinstance(value, dict) else value).casefold()
            if name and (name in question_folded or name in selected_text) and entity_count < 32:
                result.add(node_id)
                entity_count += 1
            continue
        result.add(node_id)
    return result


def build_context_slice(
    store: GraphStore,
    *,
    task_id: str,
    branch_id: str,
    role: str,
    root_node_ids: Optional[Iterable[str]] = None,
    policy: str = "dependency_closure",
    token_budget: Optional[int] = None,
    allow_missing: bool = False,
    visible_fragment_ids: Optional[Iterable[str]] = None,
    multihop_sentence_limit: Optional[int] = None,
) -> ContextSlice:
    # External workflow adapters, such as AFlow, have their own receiver
    # contracts rather than one of the native planner/solver/critic roles.
    # They still use this same slicer; the adapter supplies deterministic roots
    # after resolving its contract.  The native role-based path remains the
    # default for all existing callers.
    if root_node_ids is None:
        roots, missing = _required_roots(store, task_id, branch_id, role)
    else:
        state = store.snapshot()
        roots = []
        missing = []
        for raw_node_id in dict.fromkeys(str(item) for item in root_node_ids):
            node = state.nodes.get(raw_node_id)
            if node is None:
                missing.append(raw_node_id)
                continue
            if node.task_id != task_id or node.branch_id != branch_id:
                raise SliceError(
                    f"context root {raw_node_id} belongs to a different task or branch"
                )
            if node.is_operationally_valid():
                roots.append(raw_node_id)
        if missing and not allow_missing:
            raise SliceError(f"context roots missing for {role}: {', '.join(missing)}")
    if missing and not allow_missing:
        raise SliceError(f"semantic context missing for {role}: {', '.join(missing)}")

    if policy in {"planner_state", "solver_state"}:
        # A planner emits several incremental Actions.  It must see the
        # Actions already committed in this stage; a dependency closure rooted
        # only at the immutable task node would make it repeat the same Action.
        visible_ids = {
            node.node_id
            for node in store.snapshot().nodes.values()
            if node.task_id == task_id
            and node.branch_id == branch_id
            and node.is_operationally_valid()
        }
    elif policy == "role_subscription":
        required = _required_logical_ids(store, task_id, branch_id, role)
        visible_ids = {
            node.node_id
            for node in store.snapshot().nodes.values()
            if node.task_id == task_id
            and node.branch_id == branch_id
            and any(node.logical_id == item or node.logical_id.startswith(item + "_") for item in required)
            and node.is_operationally_valid()
        }
    elif policy == "dependency_closure":
        visible_ids = upstream_closure(store, roots)
    elif policy == "minimal_verified":
        if role != "final_solver":
            raise SliceError("minimal_verified is only defined for final_solver")
        result = store.latest_valid(task_id, branch_id, "result")
        verification = store.latest_valid(task_id, branch_id, "verification")
        if result is None or verification is None:
            raise SliceError("minimal_verified requires current result and verification")
        if verification.status != "verified" or not verification.is_operationally_valid():
            raise SliceError("minimal_verified requires a current verified verification node")
        state = store.snapshot()
        verifies_current_result = any(
            edge.source == verification.node_id
            and edge.target == result.node_id
            and edge.relation == "verifies"
            for edge in state.edges
        )
        if not verifies_current_result:
            raise SliceError("current verification does not validate current result version")
        if not result.is_semantically_verified():
            raise SliceError(
                "result is operationally valid but lacks evidence_checked or model_judged_correct"
            )
        # The final formatter needs the current roots and their provenance, but
        # it must not expand a semantically verified result into stale history.
        visible_ids = set(roots)
    else:
        raise SliceError(f"unknown context slice policy: {policy}")

    if multihop_sentence_limit is not None:
        task_type = _task_type_from_query_or_marker(store, task_id, branch_id)
        if task_type == "multihop_qa":
            visible_ids = _multihop_source_subset(
                store,
                set(visible_ids),
                task_id=task_id,
                branch_id=branch_id,
                sentence_limit=int(multihop_sentence_limit),
            )

    state = store.snapshot()
    visible_fragments = sorted(str(item) for item in (visible_fragment_ids or ()))
    nodes = [state.nodes[node_id] for node_id in visible_ids]
    nodes.sort(key=_node_sort_key)
    local_node_id_by_node = {
        node.node_id: f"n{index}"
        for index, node in enumerate(nodes, start=1)
    }
    fragment_provenance = _fragment_provenance(nodes, visible_fragments, local_node_id_by_node)
    omitted_ids = sorted(
        node.node_id
        for node in state.nodes.values()
        if node.task_id == task_id
        and node.branch_id == branch_id
        and node.node_id not in visible_ids
        and node.is_operationally_valid()
    )

    visible_edges = [
        edge
        for edge in state.edges
        if edge.source in visible_ids and edge.target in visible_ids
    ]
    visible_edges.sort(key=lambda edge: (edge.relation, edge.source, edge.target, edge.edge_id))

    read_versions = {node.logical_id: node.node_id for node in nodes}
    digest_records = [
        {
            "slot": node.logical_id,
            "version_id": canonical_json(
                {
                    "node_id": node.node_id,
                    "content_digest": node.content_digest,
                    "status": node.status,
                    "validation": node.validation,
                }
            ),
        }
        for node in nodes
    ]
    slice_digest = dependency_digest(digest_records)
    raw_id = canonical_json(
        {
            "task_id": task_id,
            "branch_id": branch_id,
            "role": role,
            "policy": policy,
            "roots": roots,
            "read_versions": read_versions,
        }
    )
    slice_id = "slice_" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

    inclusion_reasons: Dict[str, str] = {}
    for node in nodes:
        if node.node_id in roots:
            inclusion_reasons[node.node_id] = "required role input"
        elif node.is_semantically_verified() and policy == "minimal_verified":
            inclusion_reasons[node.node_id] = "trusted semantic boundary"
        else:
            inclusion_reasons[node.node_id] = "upstream dependency of required input"

    rendered = render_compact_context_slice(
        ContextSlice(
            slice_id=slice_id,
            task_id=task_id,
            branch_id=branch_id,
            target_role=role,
            policy=policy,
            root_node_ids=roots,
            visible_node_ids=[node.node_id for node in nodes],
            visible_edge_ids=[edge.edge_id for edge in visible_edges],
            omitted_node_ids=omitted_ids,
            boundary_node_ids=[
                node.node_id
                for node in nodes
                if node.is_semantically_verified() and policy == "minimal_verified"
            ],
            read_versions=read_versions,
            dependency_digest=slice_digest,
            token_count=0,
            render_mode="structured_graph",
            inclusion_reasons=inclusion_reasons,
            errors=missing,
            node_order_digest=node_order_digest([node.node_id for node in nodes]),
            visible_fragment_ids=visible_fragments,
            local_node_id_by_node=local_node_id_by_node,
            fragment_provenance=fragment_provenance,
        ),
        state,
    )
    token_count = _token_count(rendered)
    if token_budget is not None and token_count > token_budget:
        raise SliceError(
            f"context slice exceeds budget: {token_count} > {token_budget}"
        )

    result = ContextSlice(
        slice_id=slice_id,
        task_id=task_id,
        branch_id=branch_id,
        target_role=role,
        policy=policy,
        root_node_ids=roots,
        visible_node_ids=[node.node_id for node in nodes],
        visible_edge_ids=[edge.edge_id for edge in visible_edges],
        omitted_node_ids=omitted_ids,
        boundary_node_ids=[
            node.node_id
            for node in nodes
            if node.is_semantically_verified() and policy == "minimal_verified"
        ],
        read_versions=read_versions,
        dependency_digest=slice_digest,
        token_count=token_count,
        render_mode="structured_graph",
        inclusion_reasons=inclusion_reasons,
        errors=missing,
        node_order_digest=node_order_digest([node.node_id for node in nodes]),
        visible_fragment_ids=visible_fragments,
        local_node_id_by_node=local_node_id_by_node,
        fragment_provenance=fragment_provenance,
    )
    return result


def _fragment_provenance(
    nodes: list[NodeVersion],
    visible_fragment_ids: list[str],
    local_node_id_by_node: dict[str, str],
) -> dict[str, dict[str, Any]]:
    visible = set(visible_fragment_ids)
    provenance: dict[str, dict[str, Any]] = {}
    for node in nodes:
        for fragment in fragments_for_node(node, node_scope="MANDATORY_STAGE"):
            if visible and fragment.fragment_id not in visible:
                continue
            record = {
                "fragment_id": fragment.fragment_id,
                "fragment_type": fragment.name,
                "source_node_id": node.node_id,
                "source_logical_id": node.logical_id,
                "source_node_type": node.type,
                "local_node_id": local_node_id_by_node.get(node.node_id, ""),
            }
            provenance[fragment.fragment_id] = record
            provenance[fragment.name] = record
        for fragment in fragments_for_node(node, node_scope="OPTIONAL_POLICY"):
            if visible and fragment.fragment_id not in visible:
                continue
            record = {
                "fragment_id": fragment.fragment_id,
                "fragment_type": fragment.name,
                "source_node_id": node.node_id,
                "source_logical_id": node.logical_id,
                "source_node_type": node.type,
                "local_node_id": local_node_id_by_node.get(node.node_id, ""),
            }
            provenance[fragment.fragment_id] = record
            provenance.setdefault(fragment.name, record)
    for node in nodes:
        local_id = local_node_id_by_node.get(node.node_id)
        if local_id:
            provenance[local_id] = {
                "fragment_id": "",
                "fragment_type": "node",
                "source_node_id": node.node_id,
                "source_logical_id": node.logical_id,
                "source_node_type": node.type,
                "local_node_id": local_id,
            }
    return provenance


def _content_for_render(node: NodeVersion) -> str:
    return json.dumps(node.content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_context_slice(context_slice: ContextSlice, state: Optional[GraphState] = None) -> str:
    if state is None:
        raise SliceError("render_context_slice requires a graph state")
    nodes = [state.nodes[node_id] for node_id in context_slice.visible_node_ids]
    nodes.sort(key=_node_sort_key)
    visible_edge_ids = set(context_slice.visible_edge_ids)
    edges = [edge for edge in state.edges if edge.edge_id in visible_edge_ids]
    edges.sort(key=lambda edge: (edge.relation, edge.source, edge.target, edge.edge_id))
    fragments_by_node: dict[str, set[str]] = {}
    for fragment_id in context_slice.visible_fragment_ids:
        node_id = str(fragment_id).split("#", 1)[0]
        fragments_by_node.setdefault(node_id, set()).add(str(fragment_id))

    lines = [
        "<WORKFLOW_GRAPH>",
        f"<META task_id={json.dumps(context_slice.task_id)} branch_id={json.dumps(context_slice.branch_id)} role={json.dumps(context_slice.target_role)} policy={json.dumps(context_slice.policy)}>",
    ]
    for node in nodes:
        content_value = render_fragmented_content(node, fragments_by_node.get(node.node_id, set()))
        lines.extend(
            [
                f"<NODE id={json.dumps(node.node_id)} logical_id={json.dumps(node.logical_id)} type={json.dumps(node.type)} version={node.version} status={json.dumps(node.status)}>",
                f"<CONTENT>{json.dumps(content_value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}</CONTENT>",
                f"<DEPENDENCIES>{json.dumps(node.dependency_versions, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}</DEPENDENCIES>",
                f"<VALIDATION>{json.dumps(node.validation, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}</VALIDATION>",
                f"<SOURCES>{json.dumps(node.source_refs, ensure_ascii=False, separators=(',', ':'))}</SOURCES>",
                "</NODE>",
            ]
        )
    for edge in edges:
        lines.append(
            f"<EDGE id={json.dumps(edge.edge_id)} source={json.dumps(edge.source)} relation={json.dumps(edge.relation)} target={json.dumps(edge.target)}/>")
    lines.append("</META>")
    lines.append("</WORKFLOW_GRAPH>")
    return "\n".join(lines)


def render_compact_context_slice(context_slice: ContextSlice, state: GraphState) -> str:
    """Render the selected runtime subgraph with stable local ids and short tags."""
    token_by_type = {
        "task": "T",
        "query_spec": "Q",
        "facts": "F",
        "fact": "F",
        "plan": "P",
        "plan_steps": "PS",
        "calculation": "C",
        "result": "R",
        "verification": "V",
        "final_answer": "A",
    }
    local_ids = {
        node_id: f"n{index}"
        for index, node_id in enumerate(context_slice.visible_node_ids, start=1)
    }
    lines = ["<G>"]
    nodes = [state.nodes[node_id] for node_id in context_slice.visible_node_ids]
    nodes.sort(key=_node_sort_key)
    fragments_by_node: dict[str, set[str]] = {}
    for fragment_id in context_slice.visible_fragment_ids:
        node_id = str(fragment_id).split("#", 1)[0]
        fragments_by_node.setdefault(node_id, set()).add(str(fragment_id))
    local_ids = {
        node.node_id: f"n{index}"
        for index, node in enumerate(nodes, start=1)
    }
    for node in nodes:
        tag = token_by_type.get(node.type, "N")
        content_value = render_fragmented_content(node, fragments_by_node.get(node.node_id, set()))
        if node.type == "supporting_fact":
            if isinstance(content_value, dict):
                content_value = {
                    "citation_id": node.logical_id,
                    "local_node_id": local_ids[node.node_id],
                    **content_value,
                }
            else:
                content_value = {
                    "citation_id": node.logical_id,
                    "local_node_id": local_ids[node.node_id],
                    "text": content_value,
                }
        content = json.dumps(content_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append(
            f"<{tag} id={local_ids[node.node_id]} v={node.version} s={node.status}>{content}</{tag}>"
        )
    visible_edges = set(context_slice.visible_edge_ids)
    for edge in state.edges:
        if edge.edge_id not in visible_edges:
            continue
        lines.append(
            f"<E a={local_ids[edge.source]} r={edge.relation} b={local_ids[edge.target]}/>"
        )
    lines.append("</G>")
    return "\n".join(lines)


def render_full_graph_context(
    store: GraphStore,
    *,
    task_id: str,
    branch_id: str,
    role: str,
) -> str:
    state = store.snapshot()
    latest_nodes: Dict[str, NodeVersion] = {}
    for node in state.nodes.values():
        latest = store.latest_valid(task_id, branch_id, node.logical_id)
        if latest is not None:
            latest_nodes[latest.node_id] = latest
    nodes = sorted(latest_nodes.values(), key=_node_sort_key)
    visible_ids = {node.node_id for node in nodes}
    visible_edges = [
        edge.edge_id
        for edge in state.edges
        if edge.source in visible_ids and edge.target in visible_ids
    ]
    read_versions = {node.logical_id: node.node_id for node in nodes}
    context_slice = ContextSlice(
        slice_id="full_graph",
        task_id=task_id,
        branch_id=branch_id,
        target_role=role,
        policy="full_graph",
        root_node_ids=[node.node_id for node in nodes],
        visible_node_ids=[node.node_id for node in nodes],
        visible_edge_ids=visible_edges,
        omitted_node_ids=[],
        boundary_node_ids=[],
        read_versions=read_versions,
        dependency_digest=dependency_digest(
            [{"slot": node.logical_id, "version_id": node.node_id} for node in nodes]
        ),
        token_count=0,
        render_mode="compact_graph",
        node_order_digest=node_order_digest([node.node_id for node in nodes]),
    )
    return render_compact_context_slice(context_slice, state)
