"""Receiver-contract routing for dynamic AFlow artifacts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from .aflow_artifact_graph import ArtifactRouteResult, ArtifactSlice, ClosureReplayResult, SelectionResolution
from .aflow_artifact_dependencies import DATA_DEPENDENCY_RELATIONS, capture_invocation_input_bindings


def _normalize_code(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def resolve_ensemble_selection(output: Any, candidate_batch: Any, session: Any) -> SelectionResolution:
    ids = tuple(getattr(candidate_batch, "occurrence_ids", ()) or ())
    selected = output.get("selected_occurrence_ids") if isinstance(output, Mapping) else None
    if isinstance(selected, str):
        selected = (selected,)
    if selected:
        selected = tuple(x for x in selected if x in ids)
        if selected:
            return SelectionResolution(selected, "explicit_occurrence", False, candidate_batch.batch_id)
    value = output.get("response", output.get("code", output)) if isinstance(output, Mapping) else output
    matches = []
    for occurrence_id in ids:
        occurrence = getattr(session, "_artifact_occurrences", {}).get(occurrence_id)
        record = getattr(session, "_artifact_records", {}).get(getattr(occurrence, "canonical_artifact_id", ""))
        if record is not None and _normalize_code(record.content) == _normalize_code(value):
            matches.append(occurrence_id)
    if len(matches) == 1:
        return SelectionResolution(tuple(matches), "exact_content", False, candidate_batch.batch_id)
    if len(matches) > 1:
        return SelectionResolution(tuple(matches), "exact_duplicate_group", False, candidate_batch.batch_id)
    return SelectionResolution((), "unresolved", True, candidate_batch.batch_id)


def replay_receiver_closure(receiver: str, closure: Any, contract: Any,
                            original_kwargs: Mapping[str, Any],
                            rendered_kwargs: Mapping[str, Any] | None = None,
                            binding_valid: bool = False) -> ClosureReplayResult:
    # Replay is an independent reconstruction.  Falling back to original
    # kwargs here would make a missing closure look like a successful replay.
    rendered = dict(rendered_kwargs or {})
    receiver_name = str(receiver).casefold()
    if receiver_name == "scensemble":
        semantic = isinstance(rendered.get("solutions"), (list, tuple)) and bool(rendered.get("solutions"))
    elif receiver_name == "test":
        semantic = bool(rendered.get("solution")) and bool(rendered.get("problem"))
    elif receiver_name == "custom" and "Failed solution:" in str(original_kwargs.get("input", "")):
        semantic = bool(rendered.get("input") or rendered.get("solution"))
    else:
        semantic = bool(rendered)
    complete = bool(closure is not None and getattr(closure, "complete", False))
    replayable = complete and binding_valid and semantic
    return ClosureReplayResult(
        replayable=replayable, rendered_kwargs=rendered,
        required_fields_present=semantic, artifact_bindings_valid=complete,
        source_bindings_valid=complete, multiplicity_preserved=complete,
        semantic_equivalent=semantic,
        failure_reason="" if replayable else ("binding_unresolved" if not binding_valid else "closure_not_replayable"),
    )


def build_sendall_artifact_slice(session: Any, native_plan: Any, contract: Any) -> ArtifactSlice:
    """Return the receiver's complete visible occurrence universe."""
    batch_id = getattr(native_plan, "candidate_batch_id", None)
    occurrences = getattr(session, "_artifact_occurrences", {})
    selected = tuple(
        occurrence_id for occurrence_id, occurrence in occurrences.items()
        if occurrence.task_id == getattr(session, "task_id", occurrence.task_id)
        and (batch_id is None or occurrence.candidate_batch_id in {None, batch_id})
    )
    tokens = sum(session._count_tokens(occurrences[x].raw_value) for x in selected)
    canonical = tuple(dict.fromkeys(occurrences[x].canonical_artifact_id for x in selected))
    return ArtifactSlice(occurrence_ids=selected, canonical_node_ids=canonical,
                         transport_tokens=tokens, complete=True)


def build_minctx_artifact_slice(session: Any, native_plan: Any, sendall_slice: ArtifactSlice, contract: Any, closure_occurrence_ids: tuple[str, ...] = ()) -> ArtifactSlice:
    sendall = set(sendall_slice.occurrence_ids)
    roots = tuple(getattr(native_plan, "mandatory_root_ids", ()))
    native_selected = tuple(getattr(native_plan, "selected_occurrence_ids", ()))
    if not native_selected:
        native_selected = tuple(
            occurrence_id
            for binding in getattr(native_plan, "field_bindings", {}).values()
            for occurrence_id in getattr(binding, "occurrence_ids", ())
        )
    if not native_selected:
        canonical_roots = set(getattr(native_plan, "mandatory_root_ids", ()) or ()) | set()
        native_selected = tuple(
            occurrence_id for occurrence_id, occurrence in getattr(session, "_artifact_occurrences", {}).items()
            if occurrence.canonical_artifact_id in canonical_roots
        )
    selected = tuple(dict.fromkeys((*native_selected, *closure_occurrence_ids)))
    if not selected:
        selected = tuple(x for x in roots if x in sendall)
    if not set(selected).issubset(sendall):
        return ArtifactSlice(selected, transport_tokens=0, complete=False, failure_reason="sendall_universe_incomplete")
    occurrences = getattr(session, "_artifact_occurrences", {})
    tokens = sum(session._count_tokens(occurrences[x].raw_value) for x in selected if x in occurrences)
    canonical = tuple(dict.fromkeys(occurrences[x].canonical_artifact_id for x in selected if x in occurrences))
    return ArtifactSlice(occurrence_ids=selected, canonical_node_ids=canonical,
                         transport_tokens=tokens, complete=True)


@dataclass(frozen=True)
class ArtifactClosure:
    root_ids: tuple[str, ...] = ()
    selected_occurrence_ids: tuple[str, ...] = ()
    selected_canonical_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    dependency_edge_ids: tuple[str, ...] = ()
    hop_by_node: Mapping[str, int] = None
    unresolved_dependencies: tuple[str, ...] = ()
    complete: bool = False
    replayable: bool = False


def build_receiver_artifact_closure(
    session: Any,
    receiver: str,
    mandatory_root_ids: tuple[str, ...],
    allowed_relations: tuple[str, ...] = (),
    task_id: str | None = None,
    candidate_batch_id: str | None = None,
    max_hops: int = 4,
) -> ArtifactClosure:
    """Traverse the independent runtime dependency index, not GraphStore DAG edges."""
    roots = tuple(str(x) for x in mandatory_root_ids)
    allowed = set(allowed_relations) or set(DATA_DEPENDENCY_RELATIONS)
    edges = tuple(getattr(session, "_artifact_dependency_edges", {}).values())
    queue = [(root, 0) for root in roots]
    seen = set(roots)
    selected = list(roots)
    edge_ids: list[str] = []
    source_ids: list[str] = []
    hops: dict[str, int] = {root: 0 for root in roots}
    unresolved: list[str] = []
    while queue:
        dependent, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        for edge in edges:
            dependent_matches = edge.dependent_node_id == dependent
            if not dependent_matches and dependent in getattr(session, "_artifact_records", {}):
                occ = getattr(session, "_artifact_occurrences", {}).get(edge.dependent_node_id)
                dependent_matches = bool(occ and occ.canonical_artifact_id == dependent)
            if not dependent_matches or not edge.hard:
                continue
            if edge.relation not in allowed:
                continue
            if task_id is not None and edge.task_id != task_id:
                continue
            if candidate_batch_id is not None and edge.candidate_batch_id not in {None, candidate_batch_id}:
                continue
            prerequisite = edge.prerequisite_node_id
            edge_ids.append(edge.edge_id)
            if prerequisite not in seen:
                seen.add(prerequisite); selected.append(prerequisite)
                hops[prerequisite] = hop + 1
                queue.append((prerequisite, hop + 1))
            if (
                prerequisite.startswith("source:")
                or prerequisite in getattr(session, "_hotpot_source_node_ids", ())
                or prerequisite in getattr(session.store.snapshot(), "nodes", {})
            ):
                source_ids.append(prerequisite)
    occurrences = getattr(session, "_artifact_occurrences", {})
    canonical = tuple(dict.fromkeys(
        occurrences[node].canonical_artifact_id for node in selected if node in occurrences
    ))
    selected_occurrences = tuple(node for node in selected if node in occurrences)
    complete = not unresolved and bool(roots)
    return ArtifactClosure(
        root_ids=roots, selected_occurrence_ids=selected_occurrences,
        selected_canonical_ids=canonical, source_node_ids=tuple(dict.fromkeys(source_ids)),
        dependency_edge_ids=tuple(dict.fromkeys(edge_ids)), hop_by_node=hops,
        unresolved_dependencies=tuple(unresolved), complete=complete,
        replayable=complete,
    )


def route_receiver_context(session: Any, receiver: str, original_kwargs: Mapping[str, Any], contract: Any,
                           *, policy: str = "contract_minimal") -> ArtifactRouteResult:
    started = time.perf_counter()
    original = dict(original_kwargs)
    original_tokens = session._count_tokens(original)
    lowered = str(receiver).casefold()
    role = session.profile.operator_role(receiver, original_kwargs) if session.profile is not None else "passthrough"
    if role == "source_producer" or (role == "passthrough" and not getattr(contract, "required_artifact_types", ())):
        return ArtifactRouteResult(original, original_input_tokens=original_tokens,
                                   selected_input_tokens=original_tokens, contract_satisfied=True,
                                   closure_replayable=False,
                                   compression_reason="source_producer_passthrough",
                                   selection_overhead_ms=(time.perf_counter() - started) * 1000.0)
    selector = getattr(session.profile, "source_selector", None) if session.profile is not None else None
    roots = tuple(selector(session=session, operator=receiver, original_kwargs=original) if selector else ())
    if selector is None and not roots:
        roots = tuple(session.candidate_pool_ids() if lowered == "scensemble" else session.latest_output_ids())
    if not roots:
        return ArtifactRouteResult(original, original_input_tokens=original_tokens,
                                   selected_input_tokens=original_tokens, fallback=True,
                                   fallback_reason="missing_artifact_roots",
                                   selection_overhead_ms=(time.perf_counter() - started) * 1000.0)
    selected = roots
    closure = []
    dependency_edges = []
    closure_result = None
    native_candidates = [
        plan for plan in getattr(session, "_native_invocation_plans", {}).values()
        if plan.operator.casefold() == lowered
    ]
    native_plan = native_candidates[-1] if native_candidates else None
    required_validation = (
        session.validate_required_bindings(native_plan, contract)
        if native_plan is not None and hasattr(session, "validate_required_bindings")
        else None
    )
    if native_plan is not None and getattr(native_plan, "mandatory_root_ids", ()):
        roots = tuple(native_plan.mandatory_root_ids)
    sendall_slice = build_sendall_artifact_slice(session, native_plan or type("Plan", (), {"candidate_batch_id": getattr(session, "_last_closed_candidate_batch_id", None)})(), contract)
    if getattr(contract, "dependency_closure", True):
        allowed = set(getattr(contract, "required_relations", ()) or ())
        closure_result = build_receiver_artifact_closure(
            session, receiver, roots, tuple(allowed),
            task_id=getattr(session, "task_id", None),
            candidate_batch_id=getattr(native_plan, "candidate_batch_id", None),
        )
        closure = list(closure_result.selected_canonical_ids)
        dependency_edges = [
            {"edge_id": edge_id} for edge_id in closure_result.dependency_edge_ids
        ]
        if policy != "send_all":
            selected = tuple(dict.fromkeys((*roots, *closure)))
            minctx_slice = build_minctx_artifact_slice(
                session, native_plan or type("Plan", (), {"mandatory_root_ids": roots, "selected_occurrence_ids": ()})(),
                sendall_slice, contract,
                tuple(getattr(closure_result, "selected_occurrence_ids", ())) if closure_result else (),
            )
            if not minctx_slice.complete:
                return ArtifactRouteResult(
                    original, selected_artifact_ids=selected,
                    original_input_tokens=original_tokens,
                    selected_input_tokens=original_tokens,
                    contract_satisfied=False, fallback=True,
                    fallback_reason=minctx_slice.failure_reason,
                    selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
                )
            occurrence_map = getattr(session, "_artifact_occurrences", {})
            selected = tuple(dict.fromkeys(
                roots + tuple(occurrence_map[x].canonical_artifact_id for x in minctx_slice.occurrence_ids if x in occurrence_map)
            ))
    if policy == "send_all":
        occurrence_map = getattr(session, "_artifact_occurrences", {})
        selected = tuple(dict.fromkeys(
            roots + tuple(occurrence_map[x].canonical_artifact_id for x in sendall_slice.occurrence_ids)
        ))
    # Keep runtime identity domains separate: occurrence IDs are used for
    # binding/replay, canonical IDs only for GraphStore delivery, and source
    # IDs only for source payloads.
    occurrence_map = getattr(session, "_artifact_occurrences", {})
    selected_occurrence_ids = tuple(dict.fromkeys(
        tuple(getattr(native_plan, "mandatory_occurrence_ids", ()))
        + tuple(getattr(closure_result, "selected_occurrence_ids", ()))
    ))
    if not selected_occurrence_ids and native_plan is None:
        selected_occurrence_ids = tuple(
            oid for oid, occurrence in occurrence_map.items()
            if occurrence.canonical_artifact_id in set(roots)
        )
    selected_source_ids = tuple(dict.fromkeys(
        tuple(getattr(native_plan, "mandatory_source_ids", ()))
        + tuple(getattr(closure_result, "source_node_ids", ()))
    ))
    selected_canonical_ids = tuple(dict.fromkeys(
        tuple(occurrence_map[x].canonical_artifact_id for x in selected_occurrence_ids if x in occurrence_map)
    ))
    selected = selected_canonical_ids
    delivered = session.delivered_artifacts(selected_canonical_ids)
    if lowered == "scensemble":
        bound = session.profile.input_binding(session=session, operator=receiver,
            original_kwargs=original, delivered_artifacts=delivered)
        if bool(getattr(contract, "preserve_multiplicity", True)) and "solutions" in bound.kwargs:
            expanded = []
            for item in delivered:
                expanded.extend([item.value] * max(1, int(item.multiplicity)))
            bound.kwargs["solutions"] = expanded
    elif lowered == "custom":
        bound = session.profile.input_binding(session=session, operator=receiver,
            original_kwargs=original, delivered_artifacts=delivered)
    else:
        bound = session.profile.input_binding(session=session, operator=receiver,
            original_kwargs=original, delivered_artifacts=delivered)
    selected_tokens = session._count_tokens(bound.kwargs)
    selected_occurrences = selected_occurrence_ids
    binding_hints = {}
    if lowered == "scensemble":
        binding_hints["solutions"] = selected_occurrences
    elif lowered in {"test", "custom"} and selected_occurrences:
        binding_hints["solution"] = selected_occurrences[:1]
        if lowered == "custom" and "Failed solution:" in str(original.get("input", "")):
            binding_hints["failed_code"] = selected_occurrences[:1]
            binding_hints["test_result"] = selected_occurrences[:1]
    binding_records = capture_invocation_input_bindings(
        task_id=getattr(session, "task_id", ""), operator=receiver,
        invocation_id=f"route:{receiver}", kwargs=original,
        artifact_binding_hints=binding_hints,
        source_node_ids=selected_source_ids, candidate_batch_id=getattr(native_plan, "candidate_batch_id", None),
    )
    binding_valid = bool(required_validation.valid) if required_validation is not None else all(
        session.validate_exact_binding(binding, original).valid
        for binding in binding_records if binding.artifact_occurrence_ids
    )
    replay = replay_receiver_closure(
        receiver, closure_result, contract, original, bound.kwargs,
        binding_valid=binding_valid,
    )
    required = tuple(getattr(contract, "required_artifact_types", ()))
    forbidden = set(getattr(contract, "forbidden_artifact_types", ()) or ())
    kinds = {str(session.store.node(node_id).provenance.get("artifact_kind", "")) for node_id in selected if node_id in session.store.snapshot().nodes}
    contract_ok = set(required).issubset(kinds) and not (forbidden & kinds)
    if lowered == "test" and "solution" in original and delivered:
        contract_ok = "opaque_text" not in kinds
    if lowered == "custom" and "Failed solution:" in str(original.get("input", "")) and "Error:" in str(original.get("input", "")) and delivered:
        contract_ok = "opaque_text" not in kinds
    if not required and not forbidden:
        contract_ok = True
    if selected_tokens > int(getattr(contract, "max_tokens", 0) or 0) > 0:
        contract_ok = False
    if not binding_valid:
        return ArtifactRouteResult(original, selected_artifact_ids=selected,
            closure_artifact_ids=tuple(closure), original_input_tokens=original_tokens,
            selected_input_tokens=original_tokens, contract_satisfied=False, fallback=True,
            fallback_reason="binding_unresolved",
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
            transport_sendall_tokens=sendall_slice.transport_tokens,
            transport_minctx_tokens=sendall_slice.transport_tokens if policy == "send_all" else 0)
    if not replay.replayable:
        return ArtifactRouteResult(original, selected_artifact_ids=selected,
            closure_artifact_ids=tuple(closure), original_input_tokens=original_tokens,
            selected_input_tokens=original_tokens, contract_satisfied=False, fallback=True,
            fallback_reason=replay.failure_reason or "closure_not_replayable",
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
            transport_sendall_tokens=sendall_slice.transport_tokens,
            transport_minctx_tokens=sendall_slice.transport_tokens if policy == "send_all" else 0)
    if not contract_ok:
        return ArtifactRouteResult(original, selected_artifact_ids=selected,
            original_input_tokens=original_tokens, selected_input_tokens=original_tokens,
            contract_satisfied=False, fallback=True, fallback_reason="contract_failed",
            selection_overhead_ms=(time.perf_counter() - started) * 1000.0)
    if selected_tokens > original_tokens:
        return ArtifactRouteResult(original, selected_artifact_ids=selected,
            closure_artifact_ids=tuple(closure), original_input_tokens=original_tokens,
            selected_input_tokens=original_tokens, contract_satisfied=False, fallback=True,
            fallback_reason="selected_input_longer", selection_overhead_ms=(time.perf_counter() - started) * 1000.0)
    return ArtifactRouteResult(bound.kwargs, selected_artifact_ids=selected,
        closure_artifact_ids=tuple(closure), source_node_ids=selected_source_ids,
        original_input_tokens=original_tokens, selected_input_tokens=selected_tokens,
        saved_input_tokens=max(0, original_tokens - selected_tokens), contract_satisfied=True,
        selection_overhead_ms=(time.perf_counter() - started) * 1000.0,
        artifact_binding_hints=binding_hints,
        candidate_batch_id=getattr(native_plan, "candidate_batch_id", None),
        closure_replayable=replay.replayable,
        compression_reason=("compressed_model_input" if selected_tokens < original_tokens else "native_input_already_minimal"),
        transport_sendall_tokens=sendall_slice.transport_tokens,
        transport_minctx_tokens=(sendall_slice.transport_tokens if policy == "send_all" else sum(
            session._count_tokens(getattr(session, "_artifact_occurrences", {})[x].raw_value)
            for x in getattr(locals().get("minctx_slice", None), "occurrence_ids", ())
            if x in getattr(session, "_artifact_occurrences", {})
        )),
    )
