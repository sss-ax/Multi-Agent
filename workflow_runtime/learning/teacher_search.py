"""Phase 2A SEND search and Phase 2B active-information teacher selection."""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, Protocol

from ..communication_action import CommunicationAction, CommunicationActionType
from ..communication_replay import CommunicationDecisionSnapshot
from .schemas import (
    CounterfactualOutcome,
    TeacherSearchResult,
    ensure_same_current_label,
)


class RolloutCallable(Protocol):
    def __call__(self, action: CommunicationAction) -> CounterfactualOutcome: ...


def _action_key(action: CommunicationAction) -> tuple[object, ...]:
    return (
        action.action_type.value,
        tuple(sorted(action.selected_fragment_ids)),
        tuple(action.requested_semantics),
        action.query,
        action.target,
        repr(sorted(action.payload.items())),
    )


def _teacher_key(outcome: CounterfactualOutcome) -> tuple[object, ...]:
    return (
        not outcome.legal,
        outcome.final_outcome.success is not True,
        outcome.total_cost_tokens,
        outcome.communication_tokens,
        outcome.round_count,
        len(outcome.action.selected_fragment_ids),
    )


class _CachedRollouts:
    def __init__(self, runner: RolloutCallable) -> None:
        self.runner = runner
        self.values: dict[tuple[object, ...], CounterfactualOutcome] = {}

    def run(self, action: CommunicationAction) -> CounterfactualOutcome:
        key = _action_key(action)
        if key not in self.values:
            outcome = self.runner(action)
            if _action_key(outcome.action) != key:
                raise ValueError("rollout runner returned an outcome for a different action")
            self.values[key] = outcome
        return self.values[key]

    def outcomes(self) -> tuple[CounterfactualOutcome, ...]:
        return tuple(self.values.values())


class SendCounterfactualTeacher:
    """Find minimum-cost successful SEND packets without conflating risk labels."""

    def __init__(self, *, exhaustive_limit: int = 10, max_rollouts: int = 64) -> None:
        self.exhaustive_limit = max(0, int(exhaustive_limit))
        self.max_rollouts = max(3, int(max_rollouts))

    def search(
        self,
        snapshot: CommunicationDecisionSnapshot,
        runner: RolloutCallable,
    ) -> TeacherSearchResult:
        optional = tuple(sorted(
            str(fragment.get("fragment_id", ""))
            for fragment in snapshot.fragments
            if str(fragment.get("scope", "")) == "OPTIONAL_POLICY"
            and str(fragment.get("fragment_id", ""))
        ))
        cache = _CachedRollouts(runner)
        cache.run(CommunicationAction(CommunicationActionType.STOP))
        send_all = CommunicationAction(
            CommunicationActionType.SEND,
            selected_fragment_ids=optional,
        )
        cache.run(send_all)
        if len(optional) <= self.exhaustive_limit:
            self._enumerate(optional, cache)
        else:
            self._backward_eliminate(optional, cache)
        outcomes = cache.outcomes()
        ensure_same_current_label(outcomes)
        successful = tuple(
            outcome for outcome in outcomes
            if outcome.legal
            and outcome.action.action_type == CommunicationActionType.SEND
            and outcome.final_outcome.success is True
        )
        if not successful:
            return TeacherSearchResult(
                snapshot_id=snapshot.snapshot_id,
                task_id=snapshot.task_id,
                phase="2A",
                status="no_successful_send",
                teacher_outcomes=(),
                evaluated_outcomes=outcomes,
                reason="no SEND subset reached terminal task success; route to Phase 2B",
            )
        best_key = min(_teacher_key(item) for item in successful)
        teachers = tuple(item for item in successful if _teacher_key(item) == best_key)
        send_all_outcome = cache.run(send_all)
        status = "teacher_found" if send_all_outcome.final_outcome.success is True else "teacher_found_sendall_failed"
        return TeacherSearchResult(
            snapshot_id=snapshot.snapshot_id,
            task_id=snapshot.task_id,
            phase="2A",
            status=status,
            teacher_outcomes=teachers,
            evaluated_outcomes=outcomes,
        )

    def _enumerate(self, optional: tuple[str, ...], cache: _CachedRollouts) -> None:
        for size in range(len(optional) + 1):
            for subset in itertools.combinations(optional, size):
                if len(cache.values) >= self.max_rollouts:
                    return
                cache.run(CommunicationAction(
                    CommunicationActionType.SEND,
                    selected_fragment_ids=tuple(subset),
                ))

    def _backward_eliminate(self, optional: tuple[str, ...], cache: _CachedRollouts) -> None:
        current = optional
        while current and len(cache.values) < self.max_rollouts:
            trials: list[CounterfactualOutcome] = []
            for fragment_id in current:
                if len(cache.values) >= self.max_rollouts:
                    break
                subset = tuple(item for item in current if item != fragment_id)
                trials.append(cache.run(CommunicationAction(
                    CommunicationActionType.SEND,
                    selected_fragment_ids=subset,
                )))
            successful = [item for item in trials if item.legal and item.final_outcome.success is True]
            if not successful:
                break
            best = min(successful, key=_teacher_key)
            next_current = tuple(sorted(best.action.selected_fragment_ids))
            if next_current == current:
                break
            current = next_current


class InformationAcquisitionTeacher:
    """Compare STOP/SEND/REQUEST/RETRIEVE/REVISE using real environment rollouts."""

    REQUIRED_TYPES = frozenset({
        CommunicationActionType.STOP,
        CommunicationActionType.SEND,
        CommunicationActionType.REQUEST,
        CommunicationActionType.RETRIEVE,
        CommunicationActionType.REVISE,
    })

    def search(
        self,
        snapshot: CommunicationDecisionSnapshot,
        actions: Iterable[CommunicationAction],
        runner: RolloutCallable,
    ) -> TeacherSearchResult:
        unique = {_action_key(action): action for action in actions}
        action_types = {action.action_type for action in unique.values()}
        missing = self.REQUIRED_TYPES - action_types
        if missing:
            return TeacherSearchResult(
                snapshot_id=snapshot.snapshot_id,
                task_id=snapshot.task_id,
                phase="2B",
                status="incomplete_action_space",
                teacher_outcomes=(),
                evaluated_outcomes=(),
                reason=f"missing action types: {sorted(item.value for item in missing)}",
            )
        cache = _CachedRollouts(runner)
        for action in unique.values():
            cache.run(action)
        outcomes = cache.outcomes()
        ensure_same_current_label(outcomes)
        successful = tuple(
            outcome for outcome in outcomes
            if outcome.legal and outcome.final_outcome.success is True
        )
        if not successful:
            return TeacherSearchResult(
                snapshot_id=snapshot.snapshot_id,
                task_id=snapshot.task_id,
                phase="2B",
                status="no_successful_action",
                teacher_outcomes=(),
                evaluated_outcomes=outcomes,
                reason="no active-information action reached terminal success",
            )
        best_key = min(_teacher_key(item) for item in successful)
        teachers = tuple(item for item in successful if _teacher_key(item) == best_key)
        return TeacherSearchResult(
            snapshot_id=snapshot.snapshot_id,
            task_id=snapshot.task_id,
            phase="2B",
            status="teacher_found",
            teacher_outcomes=teachers,
            evaluated_outcomes=outcomes,
        )
