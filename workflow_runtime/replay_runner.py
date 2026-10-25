"""Isolated end-to-end counterfactual continuation from decision snapshots."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from .communication_action import CommunicationAction, CommunicationActionExecutor
from .communication_replay import CommunicationDecisionSnapshot
from .learning.schemas import CounterfactualOutcome, CurrentAnswerLabel, FinalOutcomeLabel


class ReplayableWorkflow(Protocol):
    store: Any
    agent_views: Any

    def restore_communication_snapshot(
        self,
        snapshot: CommunicationDecisionSnapshot,
        *,
        restore_external: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, Any]: ...


WorkflowFactory = Callable[[CommunicationDecisionSnapshot], ReplayableWorkflow]
CurrentAnswerLabeler = Callable[[CommunicationDecisionSnapshot], CurrentAnswerLabel]
Continuation = Callable[
    [ReplayableWorkflow, Mapping[str, Any], CommunicationAction],
    Mapping[str, Any],
]


class CounterfactualReplayRunner:
    """Run every action in a fresh runtime and preserve the pre-action label.

    The caller owns workflow-specific continuation because LangGraph routing,
    external retrieval, and revision semantics differ by task. The continuation
    result must contain ``final_success`` and may include token/round metrics.
    """

    def __init__(
        self,
        snapshot: CommunicationDecisionSnapshot,
        *,
        workflow_factory: WorkflowFactory,
        continuation: Continuation,
        current_answer_labeler: CurrentAnswerLabeler,
        action_executor_factory: Callable[[], CommunicationActionExecutor] = CommunicationActionExecutor,
        external_restorer_factory: Callable[[ReplayableWorkflow], Callable[[Mapping[str, Any]], None] | None]
        | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.workflow_factory = workflow_factory
        self.continuation = continuation
        self.current_answer_labeler = current_answer_labeler
        self.action_executor_factory = action_executor_factory
        self.external_restorer_factory = external_restorer_factory
        # Candidate correctness belongs to the state, not to a later action.
        self.current_answer_label = current_answer_labeler(snapshot)

    def __call__(self, action: CommunicationAction) -> CounterfactualOutcome:
        workflow = self.workflow_factory(self.snapshot)
        restore_external = (
            self.external_restorer_factory(workflow)
            if self.external_restorer_factory is not None else None
        )
        workflow_state = workflow.restore_communication_snapshot(
            self.snapshot,
            restore_external=restore_external,
        )
        executor = self.action_executor_factory()
        preview = executor.apply(
            self.snapshot,
            action,
            store=workflow.store,
            views=workflow.agent_views,
        )
        if not preview.legal:
            return CounterfactualOutcome(
                action=action,
                legal=False,
                current_answer=self.current_answer_label,
                final_outcome=FinalOutcomeLabel(
                    success=None,
                    policy_name=action.action_type.value,
                    evaluator_source="not_run",
                ),
                communication_tokens=preview.communication_tokens,
                failure_reason=preview.reason,
                metadata={"preview": preview.as_dict()},
            )
        terminal = dict(self.continuation(workflow, workflow_state, action))
        if "final_success" not in terminal:
            raise ValueError("counterfactual continuation must return final_success")
        success_value = terminal.get("final_success")
        success = None if success_value is None else bool(success_value)
        return CounterfactualOutcome(
            action=action,
            legal=True,
            current_answer=self.current_answer_label,
            final_outcome=FinalOutcomeLabel(
                success=success,
                policy_name=str(terminal.get("policy_name", action.action_type.value)),
                evaluator_source=str(terminal.get("evaluator_source", "task_evaluator")),
            ),
            communication_tokens=int(terminal.get("communication_tokens", preview.communication_tokens) or 0),
            model_input_tokens=int(terminal.get("model_input_tokens", 0) or 0),
            model_output_tokens=int(terminal.get("model_output_tokens", 0) or 0),
            round_count=int(terminal.get("round_count", 0) or 0),
            policy_cost_tokens=int(terminal.get("policy_cost_tokens", 0) or 0),
            final_answer=terminal.get("final_answer"),
            failure_reason=str(terminal.get("failure_reason", "")),
            metadata={
                "preview": preview.as_dict(),
                **dict(terminal.get("metadata", {})),
            },
        )
