"""Compile validated incremental Actions into GraphStore mutations."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .graph_store import GraphStore
from .protocol import normalize_action_payload, require_exact_copy, validate_action
from .tools import ToolError, ToolRegistry


class ActionCompilationError(ValueError):
    """Raised when a valid Action cannot be applied to the current graph."""


FINAL_VALUE_SOURCE_TYPES = frozenset({"result", "code", "final_answer"})


@dataclass(frozen=True)
class CompilationResult:
    action: dict[str, Any]
    node_ids: tuple[str, ...] = ()
    done: bool = False


class ActionCompiler:
    """Translate role-scoped Actions into deterministic graph operations."""

    def __init__(
        self,
        store: GraphStore,
        *,
        task_id: str,
        branch_id: str = "main",
        task_type: str,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.store = store
        self.task_id = task_id
        self.branch_id = branch_id
        self.task_type = task_type
        self.tool_registry = tool_registry or ToolRegistry.default()
        self._node_ref_context: dict[str, Any] = {}

    def set_node_ref_context(self, context: dict[str, Any] | None) -> None:
        self._node_ref_context = copy.deepcopy(context or {})

    def apply(self, role: str, action: dict[str, Any]) -> CompilationResult:
        action = normalize_action_payload(action)
        errors = validate_action(role, action, task_type=self.task_type)
        if errors:
            raise ActionCompilationError(f"invalid {role} Action: {'; '.join(errors)}")
        op = action["op"]
        if op == "done":
            return CompilationResult(action=copy.deepcopy(action), done=True)

        handler = getattr(self, f"_compile_{op}", None)
        if handler is None:
            raise ActionCompilationError(f"no compiler for Action {op}")
        node_ids = tuple(handler(role, action))
        return CompilationResult(action=copy.deepcopy(action), node_ids=node_ids)

    def _add_node(
        self,
        role: str,
        *,
        logical_id: str,
        node_type: str,
        content: Any,
        status: str = "ready",
        provenance: dict[str, Any] | None = None,
    ) -> str:
        node = self.store.add_node(
            task_id=self.task_id,
            branch_id=self.branch_id,
            logical_id=logical_id,
            node_type=node_type,
            content=content,
            owner=role,
            status=status,
            validation={"schema_valid": True},
            created_by_role=role,
            provenance=provenance or {},
        )
        return node.node_id

    def _append_aggregate(self, role: str, *, logical_id: str, node_type: str, item: Any) -> str:
        previous = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
        values: list[Any]
        if previous is None:
            values = []
        elif isinstance(previous.content, list):
            values = copy.deepcopy(previous.content)
        else:
            values = [copy.deepcopy(previous.content)]
        values.append(copy.deepcopy(item))
        return self._add_node(role, logical_id=logical_id, node_type=node_type, content=values)

    def _latest_any(self, *logical_ids: str) -> str | None:
        for logical_id in logical_ids:
            node = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
            if node is not None:
                return node.node_id
        return None

    def _latest_node_any(self, *logical_ids: str):
        for logical_id in logical_ids:
            node = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
            if node is not None:
                return node
        return None

    def _code_answer_source(self):
        return self._latest_node_any("code", "verified_solution", "solver_output", "result", "execution")

    def _code_answer_value(self, source: Any) -> Any:
        content = source.content
        if source.logical_id == "code" or source.type == "code":
            return content
        if isinstance(content, dict):
            if isinstance(content.get("code"), str):
                return content["code"]
            value = content.get("value")
            if isinstance(value, dict) and isinstance(value.get("code"), str):
                return value["code"]
            if isinstance(value, str) and source.logical_id in {"code", "verified_solution", "solver_output"}:
                return value
        return content.get("value") if isinstance(content, dict) and "value" in content else content

    def _latest_with_prefix(self, prefix: str):
        candidates = [
            node for node in self.store.snapshot().nodes.values()
            if node.task_id == self.task_id
            and node.branch_id == self.branch_id
            and node.logical_id.startswith(prefix)
            and node.is_operationally_valid()
        ]
        return max(candidates, key=lambda node: (node.version, node.created_at, node.node_id), default=None)

    def _same_latest_content(self, logical_id: str, content: Any) -> bool:
        node = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
        return node is not None and node.content == content

    def _aggregate_contains(self, logical_id: str, item: Any) -> bool:
        node = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
        if node is None:
            return False
        values = node.content if isinstance(node.content, list) else [node.content]
        return any(value == item for value in values)

    def _resolve_node_ref(self, ref: Any, *, purpose: str):
        logical_id = ""
        expected_type = ""
        version = None
        if isinstance(ref, dict):
            logical_id = str(ref.get("logical_id", "")).strip()
            expected_type = str(ref.get("expected_type", ref.get("type", ""))).strip()
            version = ref.get("version")
        else:
            logical_id = str(ref).strip()

        context_node = self._resolve_node_ref_from_context(logical_id)
        if context_node is not None:
            return context_node

        aliases = {
            "answer": "result",
            "final_answer": "result",
            "final": "result",
        }
        if self.task_type == "code_generation":
            aliases.update({"answer": "code", "final_answer": "code", "final": "code", "result": "code"})
        logical_id = aliases.get(logical_id, logical_id)

        context_node = self._resolve_node_ref_from_context(logical_id)
        if context_node is not None:
            return context_node

        if re.fullmatch(r"n\d+", logical_id) or re.fullmatch(r"R\d+", logical_id):
            return None

        if expected_type and not logical_id:
            node = self._latest_with_prefix(expected_type + "_") or self._latest_node_any(expected_type)
            if node is not None:
                return node

        if logical_id:
            node = self.store.latest_valid(self.task_id, self.branch_id, logical_id)
            if node is None and expected_type and not logical_id.startswith(expected_type + "_"):
                node = self.store.latest_valid(self.task_id, self.branch_id, f"{expected_type}_{logical_id}")
            if node is not None:
                if version is not None and str(node.version) != str(version):
                    raise ActionCompilationError(
                        f"{purpose} ref version mismatch: {logical_id}@v{version} is not current {node.node_id}"
                    )
                return node

        return None

    def _resolve_node_ref_from_context(self, ref: str):
        key = str(ref).strip()
        if not key:
            return None
        state = self.store.snapshot()
        if key in state.nodes:
            return state.nodes[key]
        fragment_index = self._node_ref_context.get("fragment_index", {})
        local_node_index = self._node_ref_context.get("local_node_index", {})
        semantic_type_index = self._node_ref_context.get("semantic_type_index", {})
        for index in (fragment_index, local_node_index, semantic_type_index):
            if not isinstance(index, dict):
                continue
            value = index.get(key)
            node = self._node_from_ref_index_value(value)
            if node is not None:
                return node
        return None

    def _node_from_ref_index_value(self, value: Any):
        if value is None:
            return None
        if isinstance(value, str):
            return self.store.latest_valid(self.task_id, self.branch_id, value) or self.store.snapshot().nodes.get(value)
        if isinstance(value, dict):
            node_id = str(value.get("source_node_id", value.get("node_id", ""))).strip()
            if node_id:
                node = self.store.snapshot().nodes.get(node_id)
                if node is not None:
                    return node
            logical_id = str(value.get("source_logical_id", value.get("logical_id", ""))).strip()
            if logical_id:
                return self.store.latest_valid(self.task_id, self.branch_id, logical_id)
        if isinstance(value, list) and len(value) == 1:
            return self._node_from_ref_index_value(value[0])
        return None

    def _resolve_final_value_source(self, source: Any):
        visited: set[str] = set()
        node = source
        while node is not None:
            if node.node_id in visited:
                raise ActionCompilationError(f"cyclic final answer source reference: {node.node_id}")
            visited.add(node.node_id)
            if node.type in FINAL_VALUE_SOURCE_TYPES:
                return node
            redirected = self._redirect_final_source(node)
            if redirected is None:
                raise ActionCompilationError(
                    f"invalid final answer source type: {node.type} ({node.node_id})"
                )
            node = redirected
        raise ActionCompilationError("invalid final answer source: empty redirect")

    def _redirect_final_source(self, node: Any):
        if node.type == "verification":
            target_ref = None
            if isinstance(node.content, dict):
                target_ref = node.content.get("target")
            if target_ref:
                target = self._resolve_node_ref(target_ref, purpose="verification final answer target")
                if target is not None and target.node_id != node.node_id:
                    return target
            state = self.store.snapshot()
            for edge in state.edges:
                if edge.source == node.node_id and edge.relation in {"verifies", "contradicts"}:
                    target = state.nodes.get(edge.target)
                    if target is not None:
                        return target
        return None

    def _link_if_present(self, source_logical_id: str, target_node_id: str, relation: str = "depends_on") -> None:
        source = self.store.latest_valid(self.task_id, self.branch_id, source_logical_id)
        if source is not None:
            self.store.add_edge(source=source.node_id, target=target_node_id, relation=relation, created_by_role="runtime")

    def _compile_declare_query(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        if self._same_latest_content("query_spec", action["content"]):
            return ()
        yield self._add_node(
            role,
            logical_id="query_spec",
            node_type="query_spec",
            content=action["content"],
            provenance={"communication_scope": "MANDATORY"},
        )

    def _compile_add_fact(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        item = {"id": action["id"], "value": action["value"]}
        if self._same_latest_content(f"fact_{action['id']}", item) and self._aggregate_contains("facts", item):
            return ()
        fact_id = self._add_node(role, logical_id=f"fact_{action['id']}", node_type="fact", content=item)
        facts_id = self._append_aggregate(role, logical_id="facts", node_type="facts", item=item)
        self.store.add_edge(source=fact_id, target=facts_id, relation="input_to", created_by_role=role)
        yield fact_id
        yield facts_id

    def _compile_add_evidence(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"evidence_{action['id']}", node_type="evidence", content=action["content"])

    def _compile_add_entity(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"entity_{action['id']}", node_type="entity", content=action["content"])

    def _compile_add_supporting_fact(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"supporting_fact_{action['id']}", node_type="supporting_fact", content=action["content"])

    def _compile_add_evidence_link(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"evidence_link_{action['id']}", node_type="evidence_link", content=action["content"])

    def _compile_add_table(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id="table", node_type="table", content=action["content"])

    def _compile_add_table_cell(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"table_cell_{action['id']}", node_type="table_cell", content=action["content"])

    def _compile_add_requirement(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id="requirements", node_type="requirements", content=action["content"])

    def _compile_add_test(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"test_{action['id']}", node_type="test", content=action["content"])

    def _compile_add_plan_step(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        step = {
            "id": action["id"],
            "operation": action["operation"],
            "inputs": list(action["inputs"]),
        }
        if self._same_latest_content(f"plan_step_{action['id']}", step) and self._aggregate_contains("plan_steps", step):
            return ()
        step_id = self._add_node(role, logical_id=f"plan_step_{action['id']}", node_type="plan_steps", content=step)
        self._link_if_present("query_spec", step_id, "depends_on")
        for item in action["inputs"]:
            self._link_if_present(f"fact_{item}", step_id, "depends_on")
            self._link_if_present(str(item), step_id, "depends_on")
        steps_id = self._append_aggregate(role, logical_id="plan_steps", node_type="plan_steps", item=step)
        self.store.add_edge(source=step_id, target=steps_id, relation="input_to", created_by_role=role)
        plan_id = self._append_aggregate(
            role,
            logical_id="plan",
            node_type="plan",
            item={"id": action["id"], "operation": action["operation"], "inputs": list(action["inputs"])},
        )
        self.store.add_edge(source=step_id, target=plan_id, relation="input_to", created_by_role=role)
        yield step_id
        yield steps_id
        yield plan_id

    def _compile_calculate(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        content = {
            "id": action["id"],
            "expression": action["expression"],
            "value": action["value"],
        }
        calculation_id = self._add_node(role, logical_id="calculation", node_type="calculation", content=content)
        source_id = self._latest_any(f"plan_step_{action['id']}", "plan_steps", "plan")
        if source_id is not None:
            self.store.add_edge(source=source_id, target=calculation_id, relation="derived_from", created_by_role=role)
        for logical_id in ("query_spec", "facts", "plan", "plan_steps"):
            self._link_if_present(logical_id, calculation_id, "depends_on")
        yield calculation_id

    def _compile_set_result(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        content = {"id": action["id"], "value": action["value"]}
        if self._same_latest_content("result", content):
            return ()
        result_id = self._add_node(role, logical_id="result", node_type="result", content=content)
        source_id = self._latest_any("calculation", "execution", f"test_result_{action['id']}", "query_spec")
        if source_id is not None:
            self.store.add_edge(source=source_id, target=result_id, relation="derived_from", created_by_role=role)
        yield result_id

    def _compile_emit_code(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        code_id = self._add_node(role, logical_id="code", node_type="code", content=action["code"])
        source_id = self._latest_any("requirements", "plan", "plan_steps")
        if source_id is not None:
            self.store.add_edge(source=source_id, target=code_id, relation="derived_from", created_by_role=role)
        for logical_id in ("query_spec", "requirements", "plan", "plan_steps"):
            self._link_if_present(logical_id, code_id, "depends_on")
        yield code_id

    def _compile_set_execution(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        status = "verified" if action["success"] else "need_fix"
        node_id = self._add_node(role, logical_id="execution", node_type="execution", content=action["content"], status=status)
        yield node_id

    def _compile_set_test_result(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id=f"test_result_{action['id']}", node_type="test_result", content=action["content"])

    def _compile_call_tool(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        request = {
            "call_id": action["call_id"],
            "name": action["name"],
            "arguments": copy.deepcopy(action["arguments"]),
        }
        request_id = self._add_node(
            role,
            logical_id=f"tool_request_{action['call_id']}",
            node_type="tool_request",
            content=request,
        )
        try:
            output = self.tool_registry.execute(action["name"], action["arguments"])
            result = {"call_id": action["call_id"], "name": action["name"], "ok": True, "output": output}
            status = "verified"
        except ToolError as exc:
            result = {"call_id": action["call_id"], "name": action["name"], "ok": False, "error": str(exc)}
            status = "need_fix"
        result_id = self._add_node(
            "tool",
            logical_id=f"tool_result_{action['call_id']}",
            node_type="tool_result",
            content=result,
            status=status,
        )
        self.store.add_edge(source=request_id, target=result_id, relation="produces", created_by_role="tool")
        yield request_id
        yield result_id

    def _compile_report_error(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        yield self._add_node(role, logical_id="error", node_type="error", content=action["content"], status="need_fix")

    def _compile_verify(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        verified = action["status"] == "verified"
        target = self._resolve_node_ref(action["target"], purpose="verify target")
        if target is None:
            raise ActionCompilationError(f"verify target does not exist: {action['target']}")
        target_logical_id = target.logical_id
        content = {"target": target_logical_id, "status": action["status"]}
        if not verified:
            content.update({
                "error_type": action["error_type"],
                "error_location": action["error_location"],
                "reason": action.get("reason", ""),
                "repair_instruction": action["repair_instruction"],
                "preserve": list(action.get("preserve", [])),
                "requested_fragments": list(action.get("requested_fragments", [])),
            })
        if self._same_latest_content("verification", content):
            latest_verification = self.store.latest_valid(self.task_id, self.branch_id, "verification")
            state = self.store.snapshot()
            if latest_verification is not None and any(
                edge.source == latest_verification.node_id
                and edge.target == target.node_id
                and edge.relation in {"verifies", "contradicts"}
                for edge in state.edges
            ):
                return ()
        verification_id = self._add_node(
            role,
            logical_id="verification",
            node_type="verification",
            content=content,
            status="verified" if verified else action["status"],
        )
        relation = "verifies" if verified else "contradicts"
        self.store.add_edge(source=verification_id, target=target.node_id, relation=relation, created_by_role=role)
        yield verification_id

    def _compile_answer(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        declared_source = action["source"]
        source = self._resolve_node_ref(action["source"], purpose="answer source")
        used_fallback_source = False
        if source is None and self.task_type == "code_generation":
            source = self._code_answer_source()
            used_fallback_source = source is not None
        if source is None:
            raise ActionCompilationError(f"answer source does not exist: {action['source']}")
        value_source = self._resolve_final_value_source(source)
        if self.task_type == "code_generation":
            answer_value = self._code_answer_value(value_source)
        else:
            source_value = value_source.content.get("value") if isinstance(value_source.content, dict) and "value" in value_source.content else value_source.content
            answer_value = source_value
        if self._same_latest_content("final_answer", answer_value):
            return ()
        yield self._add_node(
            role,
            logical_id="final_answer",
            node_type="final_answer",
            content=answer_value,
            status="verified",
            provenance={
                "declared_answer_source": declared_source,
                "resolved_answer_source": source.node_id,
                "answer_source_type": source.type,
                "value_answer_source": value_source.node_id,
                "value_answer_source_type": value_source.type,
                "answer_source_fallback": used_fallback_source,
            },
        )
        answer = self.store.latest_valid(self.task_id, self.branch_id, "final_answer")
        if answer is not None:
            self.store.add_edge(source=value_source.node_id, target=answer.node_id, relation="derived_from", created_by_role=role)
