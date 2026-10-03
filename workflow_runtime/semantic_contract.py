"""Structured semantic contracts for receiver-side sufficiency checks.

This module only defines contract data. Resolution is intentionally separate:
later protocol stages may decide whether a requirement is satisfied,
deterministically recoverable, or missing for a concrete receiver view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class SemanticStatus(str, Enum):
    SATISFIED = "SATISFIED"
    RECOVERABLE = "RECOVERABLE"
    MISSING = "MISSING"


@dataclass(frozen=True)
class SemanticRequirement:
    """A structured semantic object the receiver must have or recover."""

    kind: str
    target_logical_id: str | None = None
    required_type: str | None = None
    min_count: int = 1
    acceptable_logical_ids: tuple[str, ...] = ()
    acceptable_types: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("semantic requirement kind must be non-empty")
        if self.min_count < 1:
            raise ValueError("semantic requirement min_count must be positive")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "target_logical_id", _clean_optional(self.target_logical_id))
        object.__setattr__(self, "required_type", _clean_optional(self.required_type))
        object.__setattr__(self, "acceptable_logical_ids", _clean_tuple(self.acceptable_logical_ids))
        object.__setattr__(self, "acceptable_types", _clean_tuple(self.acceptable_types))

    def candidate_logical_ids(self) -> tuple[str, ...]:
        values = []
        if self.target_logical_id:
            values.append(self.target_logical_id)
        values.extend(self.acceptable_logical_ids)
        return tuple(dict.fromkeys(values))

    def candidate_types(self) -> tuple[str, ...]:
        values = []
        if self.required_type:
            values.append(self.required_type)
        values.extend(self.acceptable_types)
        return tuple(dict.fromkeys(values))

    @property
    def requirement_id(self) -> str:
        if self.target_logical_id:
            return f"{self.kind}:{self.target_logical_id}"
        if self.required_type:
            return f"{self.kind}:{self.required_type}"
        return self.kind

    @property
    def semantic_kind(self) -> str:
        return self.kind

    @property
    def target_node_id(self) -> None:
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "semantic_kind": self.semantic_kind,
            "kind": self.kind,
            "target_logical_id": self.target_logical_id,
            "target_node_id": self.target_node_id,
            "required_type": self.required_type,
            "min_count": self.min_count,
            "acceptable_logical_ids": list(self.acceptable_logical_ids),
            "acceptable_types": list(self.acceptable_types),
            "description": self.description,
        }


@dataclass(frozen=True)
class SemanticCheckResult:
    requirement: SemanticRequirement
    status: SemanticStatus
    satisfying_node_ids: tuple[str, ...] = ()
    recoverable_from_node_ids: tuple[str, ...] = ()
    recovery_method: str | None = None
    recovered_value: Any = None
    reason: str = ""

    @property
    def is_sufficient(self) -> bool:
        return self.status in {SemanticStatus.SATISFIED, SemanticStatus.RECOVERABLE}

    def as_dict(self) -> dict[str, object]:
        return {
            "requirement": self.requirement.as_dict(),
            "status": self.status.value,
            "satisfying_node_ids": list(self.satisfying_node_ids),
            "recoverable_from_node_ids": list(self.recoverable_from_node_ids),
            "recovery_method": self.recovery_method,
            "recovered_value": self.recovered_value,
            "reason": self.reason,
            "is_sufficient": self.is_sufficient,
        }


@dataclass(frozen=True)
class SemanticContract:
    sender_role: str
    receiver_role: str
    requirements: tuple[SemanticRequirement, ...] = field(default_factory=tuple)
    task_type: str = ""

    def __post_init__(self) -> None:
        if not self.receiver_role.strip():
            raise ValueError("semantic contract receiver_role must be non-empty")
        object.__setattr__(self, "sender_role", self.sender_role.strip())
        object.__setattr__(self, "receiver_role", self.receiver_role.strip())
        object.__setattr__(self, "task_type", self.task_type.strip())
        object.__setattr__(self, "requirements", tuple(self.requirements))

    @property
    def role(self) -> str:
        return self.receiver_role

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "sender_role": self.sender_role,
            "receiver_role": self.receiver_role,
            "task_type": self.task_type,
            "requirements": [requirement.as_dict() for requirement in self.requirements],
        }


def default_semantic_contract(
    sender_role: str,
    receiver_role: str,
    *,
    task_type: str = "",
) -> SemanticContract:
    """Return the default structured contract for one communication stage."""

    sender = sender_role.strip()
    receiver = receiver_role.strip()
    task = task_type.strip()
    requirements = _DEFAULT_REQUIREMENTS_BY_PAIR.get((sender, receiver), ())
    if (sender, receiver) == ("planner", "solver"):
        requirements = _planner_solver_requirements(task)
    if task == "code_generation" and (sender, receiver) == ("solver", "critic"):
        requirements = (
            SemanticRequirement(
                kind="candidate_code",
                target_logical_id="code",
                required_type="code",
                description="candidate code produced by the solver",
            ),
        )
    elif task == "code_generation" and (sender, receiver) == ("critic", "final_solver"):
        requirements = (
            SemanticRequirement(
                kind="final_candidate",
                target_logical_id="code",
                required_type="code",
                description="final code candidate",
            ),
            SemanticRequirement(
                kind="validation_signal",
                target_logical_id="verification",
                required_type="verification",
                acceptable_types=("test_result",),
                description="validation signal for the final code candidate",
            ),
        )
    return SemanticContract(
        sender_role=sender,
        receiver_role=receiver,
        task_type=task,
        requirements=requirements,
    )


def _planner_solver_requirements(task_type: str) -> tuple[SemanticRequirement, ...]:
    plan = SemanticRequirement(
        kind="plan_or_operation",
        target_logical_id="plan",
        required_type="plan",
        acceptable_logical_ids=("plan_steps",),
        acceptable_types=("plan_steps",),
        description="executable plan or operation outline for solving",
    )
    task = task_type.strip()
    if task == "multiple_choice":
        task_inputs = SemanticRequirement(
            kind="task_inputs",
            target_logical_id="choice",
            required_type="choice",
            acceptable_types=("choice",),
            description="choice options seeded for a multiple-choice solver",
        )
    elif task == "multihop_qa":
        task_inputs = SemanticRequirement(
            kind="task_inputs",
            required_type="supporting_fact",
            acceptable_types=("entity", "supporting_fact", "evidence_link", "evidence"),
            description="seeded entities and evidence for multihop reasoning",
        )
    elif task == "code_generation":
        task_inputs = SemanticRequirement(
            kind="task_inputs",
            target_logical_id="requirements",
            required_type="requirements",
            acceptable_types=("requirements", "test"),
            description="code requirements and tests seeded for the solver",
        )
    elif task == "table_qa":
        task_inputs = SemanticRequirement(
            kind="task_inputs",
            target_logical_id="table",
            required_type="table",
            acceptable_types=("table", "table_cell", "evidence"),
            description="table and evidence seeded for table QA",
        )
    else:
        task_inputs = SemanticRequirement(
            kind="task_inputs",
            target_logical_id="facts",
            required_type="facts",
            acceptable_types=("fact", "requirements", "table", "evidence"),
            description="structured task inputs needed by the solver",
        )
    return (plan, task_inputs)


def semantic_requirements_for(
    sender_role: str,
    receiver_role: str,
    *,
    task_type: str = "",
) -> tuple[SemanticRequirement, ...]:
    return default_semantic_contract(sender_role, receiver_role, task_type=task_type).requirements


def global_semantic_requirements(*, task_type: str = "") -> tuple[SemanticRequirement, ...]:
    return _GLOBAL_REQUIREMENTS


def build_role_semantic_contract(
    role: str,
    *,
    task_type: str = "",
    domain: str = "",
) -> SemanticContract:
    """Build the deterministic receiver contract for one workflow role."""

    receiver = role.strip()
    if receiver == "planner":
        requirements = global_semantic_requirements(task_type=task_type)
        sender = "global"
    else:
        sender = _ROLE_SENDER.get(receiver, "")
        requirements = default_semantic_contract(sender, receiver, task_type=task_type).requirements
    return SemanticContract(
        sender_role=sender,
        receiver_role=receiver,
        task_type=task_type,
        requirements=requirements,
    )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _clean_tuple(values: Iterable[str]) -> tuple[str, ...]:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return tuple(dict.fromkeys(cleaned))


_GLOBAL_REQUIREMENTS: tuple[SemanticRequirement, ...] = (
    SemanticRequirement(
        kind="task_goal",
        target_logical_id="task",
        required_type="task",
        description="global task goal visible before stage communication",
    ),
    SemanticRequirement(
        kind="query_spec",
        target_logical_id="query_spec",
        required_type="query_spec",
        description="global query/domain specification visible before stage communication",
    ),
)

_ROLE_SENDER: dict[str, str] = {
    "solver": "planner",
    "critic": "solver",
    "final_solver": "critic",
}

_DEFAULT_REQUIREMENTS_BY_PAIR: dict[tuple[str, str], tuple[SemanticRequirement, ...]] = {
    ("planner", "solver"): (
        SemanticRequirement(
            kind="plan_or_operation",
            target_logical_id="plan",
            required_type="plan",
            acceptable_logical_ids=("plan_steps",),
            acceptable_types=("plan_steps",),
            description="executable plan or operation outline for solving",
        ),
        SemanticRequirement(
            kind="task_inputs",
            target_logical_id="facts",
            required_type="facts",
            acceptable_types=("fact", "requirements", "table", "evidence"),
            description="structured task inputs needed by the solver",
        ),
    ),
    ("solver", "critic"): (
        SemanticRequirement(
            kind="candidate_answer_or_code",
            target_logical_id="result",
            required_type="result",
            acceptable_logical_ids=("code", "execution"),
            acceptable_types=("code", "execution", "test_result"),
            description="candidate answer artifact to be checked",
        ),
        SemanticRequirement(
            kind="support_dependencies",
            acceptable_types=("calculation", "plan_steps", "fact", "facts", "requirements", "test"),
            description="direct support needed to inspect the candidate",
        ),
    ),
    ("critic", "final_solver"): (
        SemanticRequirement(
            kind="final_candidate",
            target_logical_id="result",
            required_type="result",
            acceptable_logical_ids=("code", "final_answer"),
            acceptable_types=("code", "final_answer"),
            description="candidate answer semantics for finalization",
        ),
        SemanticRequirement(
            kind="validation_signal",
            target_logical_id="verification",
            required_type="verification",
            acceptable_types=("test_result",),
            description="critic validation signal for the final candidate",
        ),
    ),
}
