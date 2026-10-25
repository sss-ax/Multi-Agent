"""Declarative task-family contracts used by protocol and semantic compilation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping


GroundingValidator = Callable[[Any, Any], list[str]]
ResultValidator = Callable[[Any], list[str]]


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    public_source_types: frozenset[str] = frozenset({"task_source", "query_source"})
    model_node_types_by_role: Mapping[str, frozenset[str]] = field(default_factory=dict)
    allowed_relations: frozenset[str] = frozenset({"depends_on", "input_to", "derived_from", "produces", "grounded_in", "verifies", "contradicts"})
    role_context_types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    required_outputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    result_schema: Callable[[Any], list[str]] | None = None
    grounding_validator: GroundingValidator | None = None
    executor: Callable[..., Any] | None = None
    scorer: Callable[..., Any] | None = None
    planner_optional: bool = False

    def allowed_node_types(self, role: str) -> frozenset[str]:
        return frozenset(self.model_node_types_by_role.get(role, ()))

    def required_for(self, role: str) -> tuple[str, ...]:
        return tuple(self.required_outputs.get(role, ()))

    def context_for(self, role: str) -> tuple[str, ...]:
        return tuple(self.role_context_types.get(role, ()))

    def validate_result(self, value: Any) -> list[str]:
        return list(self.result_schema(value)) if self.result_schema is not None else []


def _numeric_result(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["numeric candidate_answer content must be an object"]
    if "value" not in value and "answer" not in value:
        return ["numeric candidate_answer requires value or answer"]
    return []


def _choice_result(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["multiple-choice candidate_answer content must be an object"]
    required = {"answer", "solver_choice", "slot_analysis", "eliminated_options", "confidence"}
    missing = sorted(key for key in required if key not in value)
    if missing:
        return [f"multiple-choice candidate_answer missing keys: {', '.join(missing)}"]
    slots = value.get("slot_analysis")
    if not isinstance(slots, (list, tuple)) or not slots:
        return ["multiple-choice candidate_answer slot_analysis must be a non-empty array"]
    for index, item in enumerate(slots):
        if not isinstance(item, Mapping) or set(item) != {"slot", "selected_term", "reason"}:
            return [f"multiple-choice candidate_answer slot_analysis[{index}] must contain slot, selected_term, reason"]
    eliminated = value.get("eliminated_options")
    if not isinstance(eliminated, Mapping) or not eliminated:
        return ["multiple-choice candidate_answer eliminated_options must be a non-empty object"]
    return []


def _code_result(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or not isinstance(value.get("code"), str):
        return ["code artifact content requires a code string"]
    return []


def _multihop_result(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["multihop candidate_answer content must be an object"]
    required = {"answer", "bridge_entity", "cited_fact_ids", "reasoning_chain"}
    missing = sorted(key for key in required if key not in value)
    if missing:
        return [f"multihop candidate_answer missing keys: {', '.join(missing)}"]
    if not isinstance(value.get("cited_fact_ids"), (list, tuple)) or not isinstance(value.get("reasoning_chain"), (list, tuple)):
        return ["multihop candidate_answer citations and reasoning_chain must be arrays"]
    return []


NUMERIC_SPEC = TaskSpec(
    task_type="numeric_solve",
    public_source_types=frozenset({"task_source", "query_source"}),
    model_node_types_by_role={
        "planner": frozenset({"fact", "plan_step", "plan"}),
        "solver": frozenset({"calculation", "candidate_answer"}),
        "critic": frozenset({"critique", "verification"}),
    },
    role_context_types={
        "planner": ("task_source", "query_spec"),
        "solver": ("fact", "plan_step", "plan", "calculation"),
        "critic": ("calculation", "candidate_answer", "result", "execution"),
        "final_solver": ("result", "verification"),
    },
    required_outputs={"planner": ("fact", "plan_step"), "solver": ("calculation", "candidate_answer")},
    result_schema=_numeric_result,
)

MULTIPLE_CHOICE_SPEC = TaskSpec(
    task_type="multiple_choice",
    public_source_types=frozenset({"question_source", "choice_source", "choice_schema"}),
    model_node_types_by_role={
        "planner": frozenset({"plan_step"}),
        "solver": frozenset({"slot_analysis", "candidate_answer"}),
        "critic": frozenset({"critique", "verification"}),
    },
    role_context_types={
        "planner": ("question_source", "choice_source", "choice_schema"),
        "solver": ("question_source", "choice_source", "choice_schema", "plan_step", "slot_analysis"),
        "critic": ("question_source", "choice_source", "slot_analysis", "candidate_answer", "result"),
        "final_solver": ("result", "verification"),
    },
    required_outputs={"solver": ("candidate_answer",)},
    result_schema=_choice_result,
    planner_optional=True,
)

CODE_GENERATION_SPEC = TaskSpec(
    task_type="code_generation",
    public_source_types=frozenset({"task_source", "requirements_source", "entry_point_source", "public_test_source"}),
    model_node_types_by_role={
        "planner": frozenset({"plan_step", "plan"}),
        "solver": frozenset({"code", "candidate_answer", "generated_test", "repair_patch"}),
        "critic": frozenset({"critique", "verification"}),
    },
    role_context_types={
        "planner": ("task_source", "requirements_source", "entry_point_source", "public_test_source"),
        "solver": ("requirements_source", "entry_point_source", "public_test_source", "plan_step", "plan", "repair_patch"),
        "critic": ("code", "execution", "result", "critique"),
        "final_solver": ("code", "result", "verification"),
    },
    required_outputs={"solver": ("code",)},
    result_schema=_code_result,
)

MULTIHOP_SPEC = TaskSpec(
    task_type="multihop_qa",
    public_source_types=frozenset({"question_source", "document_source", "sentence_source", "entity_source"}),
    model_node_types_by_role={
        "planner": frozenset({"entity_hypothesis", "plan_step", "retrieval_query"}),
        "solver": frozenset({"selected_evidence", "bridge_hypothesis", "reasoning_step", "candidate_answer", "retrieval_query"}),
        "critic": frozenset({"critique", "verification"}),
    },
    role_context_types={
        "planner": ("question_source", "document_source", "sentence_source", "entity_source"),
        "solver": ("sentence_source", "entity_hypothesis", "plan_step", "retrieval_query", "selected_evidence", "bridge_hypothesis", "reasoning_step"),
        "critic": ("sentence_source", "selected_evidence", "bridge_hypothesis", "reasoning_step", "candidate_answer", "result"),
        "final_solver": ("result", "verification"),
    },
    required_outputs={"solver": ("candidate_answer",)},
    result_schema=_multihop_result,
)

_GENERIC_SPECS = {
    "numeric_solve": NUMERIC_SPEC,
    "numeric_comparison": replace(NUMERIC_SPEC, task_type="numeric_comparison"),
    "multiple_choice": MULTIPLE_CHOICE_SPEC,
    "code_generation": CODE_GENERATION_SPEC,
    "multihop_qa": MULTIHOP_SPEC,
}


def get_task_spec(task_type: str) -> TaskSpec:
    """Return a registered task contract.

    Semantic Delta is fail-closed: an unregistered family has no safe node,
    relation, completion, or result contract and therefore cannot run v2.
    """
    normalized = str(task_type or "").strip()
    if normalized in _GENERIC_SPECS:
        return _GENERIC_SPECS[normalized]
    raise ValueError(f"semantic_delta_v2 does not support task_type: {normalized or 'unknown'}")


def registered_task_types() -> tuple[str, ...]:
    return tuple(sorted(_GENERIC_SPECS))
