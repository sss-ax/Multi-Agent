"""Compile validated Semantic Delta v2 Actions into GraphStore mutations."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .graph_store import GraphStore
from .protocol import validate_action
from .semantic_actions import CRITIC_TYPED_OPS, TYPED_SEMANTIC_OPS, lower_typed_semantic_action
from .semantic_delta_compiler import SemanticDeltaCompiler
from .tools import ToolRegistry
from .choice_audit import build_choice_audit_manifest


class ActionCompilationError(ValueError):
    """Raised when a valid semantic Action cannot be committed."""


@dataclass(frozen=True)
class CompilationResult:
    action: dict[str, Any]
    node_ids: tuple[str, ...] = ()
    done: bool = False
    edge_ids: tuple[str, ...] = ()
    send_hints: tuple[dict[str, Any], ...] = ()
    temp_to_node_id: dict[str, str] = field(default_factory=dict)


class ActionCompiler:
    """The deterministic boundary between model semantics and graph state."""

    def __init__(
        self,
        store: GraphStore,
        *,
        task_id: str,
        branch_id: str = "main",
        task_type: str,
        tool_registry: ToolRegistry | None = None,
        protocol_mode: str = "semantic_delta_v2",
    ) -> None:
        if protocol_mode != "semantic_delta_v2":
            raise ValueError(f"unsupported action protocol: {protocol_mode}")
        self.store = store
        self.task_id = task_id
        self.branch_id = branch_id
        self.task_type = task_type
        self.tool_registry = tool_registry or ToolRegistry.default()
        self.semantic_delta_compiler = SemanticDeltaCompiler(
            store,
            task_id=task_id,
            branch_id=branch_id,
            task_type=task_type,
        )
        self._node_ref_context: dict[str, Any] = {}

    def set_node_ref_context(self, context: dict[str, Any] | None) -> None:
        self._node_ref_context = copy.deepcopy(context or {})

    def validate_semantic(self, role: str, action: dict[str, Any]) -> None:
        try:
            lowered = lower_typed_semantic_action(role, self.task_type, action)
            self.semantic_delta_compiler.build_plan(
                role,
                lowered,
                visible_node_ids=self._node_ref_context.get("visible_node_ids", ()),
                known_fragment_ids=self._node_ref_context.get("known_fragment_ids", ()),
                allowed_receivers=self._node_ref_context.get("allowed_receivers", ()),
                node_ref_context=self._node_ref_context,
            )
        except ValueError as exc:
            raise ActionCompilationError(f"invalid Semantic Delta: {exc}") from exc

    def apply(self, role: str, action: dict[str, Any]) -> CompilationResult:
        errors = validate_action(role, action, task_type=self.task_type)
        if errors:
            raise ActionCompilationError(f"invalid {role} Action: {'; '.join(errors)}")
        op = action["op"]
        if op == "done":
            return CompilationResult(action=copy.deepcopy(action), done=True)
        if op in TYPED_SEMANTIC_OPS:
            try:
                resolved_action = self._resolve_model_action(action)
                if op == "choose_option":
                    latest = self._latest_node("result")
                    latest_value = latest.content if latest is not None else None
                    if isinstance(latest_value, dict) and "value" in latest_value:
                        latest_value = latest_value.get("value")
                    proposed_value = {
                        "answer": resolved_action["answer"],
                        "solver_choice": resolved_action["answer"],
                        "slot_analysis": resolved_action["slot_analysis"],
                        "option_reviews": resolved_action.get("option_reviews", []),
                        "eliminated_options": resolved_action["eliminated_options"],
                        "confidence": resolved_action["confidence"],
                    }
                    if latest_value == proposed_value:
                        raise ActionCompilationError(
                            "no_op_repair: revised choose_option is semantically identical to the current candidate"
                        )
                lowered = lower_typed_semantic_action(role, self.task_type, action)
                if op == "choose_option":
                    for node in lowered.get("nodes", ()):
                        if node.get("type") == "candidate_answer":
                            node_content = node.get("content")
                            if isinstance(node_content, dict):
                                node_content["slot_analysis"] = copy.deepcopy(resolved_action["slot_analysis"])
                compiled = self.semantic_delta_compiler.apply(
                    role,
                    lowered,
                    visible_node_ids=self._node_ref_context.get("visible_node_ids", ()),
                    known_fragment_ids=self._node_ref_context.get("known_fragment_ids", ()),
                    allowed_receivers=self._node_ref_context.get("allowed_receivers", ()),
                    node_ref_context=self._node_ref_context,
                )
            except ValueError as exc:
                raise ActionCompilationError(f"invalid Semantic Delta: {exc}") from exc
            return CompilationResult(
                action=copy.deepcopy(action),
                node_ids=compiled.node_ids,
                edge_ids=compiled.edge_ids,
                send_hints=compiled.send_hints,
                temp_to_node_id=compiled.temp_to_node_id,
            )
        if op in CRITIC_TYPED_OPS:
            return self.compile_critic_verification(role, action)
        if op == "repair_patch":
            if action.get("action") == "STOP":
                return CompilationResult(action=copy.deepcopy(action), done=True)
            return CompilationResult(
                action=copy.deepcopy(action),
                node_ids=tuple(self._compile_repair_patch(role, action)),
            )
        raise ActionCompilationError(f"no Semantic Delta compiler for Action {op}")

    def _resolve_model_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Resolve model slot reasons against immutable source option values."""
        if action.get("op") != "choose_option":
            return action
        answer = str(action.get("answer", "")).strip().upper()
        source_labels: set[str] = set()
        for node in self.store.snapshot().nodes.values():
            if node.task_id != self.task_id or node.branch_id != self.branch_id or node.type != "choice_source":
                continue
            raw = node.content
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {}
            if isinstance(raw, dict) and str(raw.get("label", "")).strip():
                source_labels.add(str(raw["label"]).strip().upper())
        reviews = action.get("option_reviews", ())
        review_labels = {
            str(item.get("label", "")).strip().upper()
            for item in reviews if isinstance(item, dict) and str(item.get("label", "")).strip()
        }
        if source_labels and review_labels != source_labels:
            raise ActionCompilationError(
                f"option_reviews must cover Source Graph labels exactly: expected={sorted(source_labels)}, got={sorted(review_labels)}"
            )
        selected_labels = {
            str(item.get("label", "")).strip().upper()
            for item in reviews if isinstance(item, dict) and item.get("assessment") == "selected"
        }
        if selected_labels != {answer}:
            raise ActionCompilationError("option_reviews must contain exactly the selected answer label")
        schema_node = self.store.latest_valid(self.task_id, self.branch_id, "choice_schema")
        schema: Any = {}
        if schema_node is not None:
            schema = schema_node.content
            if isinstance(schema, str):
                try:
                    schema = json.loads(schema)
                except json.JSONDecodeError:
                    schema = {}
        manifest = schema.get("slot_values", {}) if isinstance(schema, dict) else {}
        values = manifest.get(answer) if isinstance(manifest, dict) else None
        if not isinstance(values, list) or not values:
            values = []
            for node in self.store.snapshot().nodes.values():
                if node.task_id != self.task_id or node.branch_id != self.branch_id or node.type != "choice_source":
                    continue
                raw = node.content
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = {}
                if isinstance(raw, dict) and str(raw.get("label", "")).strip().upper() == answer:
                    text = str(raw.get("text", "")).strip()
                    values = [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()] or [text]
                    break
        reasons = {
            int(item["slot"]): str(item["reason"]).strip()
            for item in action.get("slot_reasons", ())
            if isinstance(item, dict) and isinstance(item.get("slot"), int)
        }
        if not values:
            raise ActionCompilationError(f"choice source for answer {answer!r} is unavailable")
        expected = set(range(1, len(values) + 1))
        if set(reasons) != expected:
            raise ActionCompilationError(
                f"slot_reasons must cover source slots exactly: expected={sorted(expected)}, got={sorted(reasons)}"
            )
        resolved = copy.deepcopy(action)
        resolved["slot_analysis"] = [
            {"slot": index, "selected_term": values[index - 1], "reason": reasons[index]}
            for index in range(1, len(values) + 1)
        ]
        resolved["eliminated_options"] = {
            str(item["label"]).strip().upper(): str(item["reason"]).strip()
            for item in action.get("option_reviews", ())
            if isinstance(item, dict) and item.get("assessment") == "rejected"
        }
        return resolved

    def compile_critic_verification(
        self,
        role: str,
        model_action: dict[str, Any],
        verification_action: dict[str, Any] | None = None,
    ) -> CompilationResult:
        errors = validate_action(role, model_action, task_type=self.task_type)
        if errors:
            raise ActionCompilationError(f"invalid {role} Action: {'; '.join(errors)}")
        internal = verification_action or self.critic_verification_action(model_action)
        return CompilationResult(
            action=copy.deepcopy(model_action),
            node_ids=tuple(self._compile_kernel_verification(role, internal)),
        )

    def critic_verification_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Lower a domain Critic Action to the kernel-private verification IR."""
        op = action["op"]
        target = "code" if self.task_type == "code_generation" else "result"
        status = "uncertain"
        error_type = ""
        error_location = ""
        reason = ""
        repair_instruction = ""
        if op == "judge_numeric":
            verdict = action["model_verdict"]
            status = {"consistent": "verified", "inconsistent": "need_fix"}.get(verdict, "uncertain")
            error_type = "" if status == "verified" else action["error_type"]
            error_location = "" if status == "verified" else action["error_location"]
            reason = action["reason"]
            repair_instruction = "" if status == "verified" else action["repair_instruction"]
        elif op == "audit_choice":
            result = self._latest_node("result")
            value = result.content if result is not None else {}
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            candidate = value if isinstance(value, dict) else {}
            candidate_slots = candidate.get("slot_analysis", [])
            candidate_eliminated = candidate.get("eliminated_options", {})
            selected_check = action.get("selected_option_check", {})
            slot_checks = action.get("slot_checks", [])
            option_checks = action.get("option_checks", [])
            manifest = build_choice_audit_manifest(self.store, task_id=self.task_id, branch_id=self.branch_id)
            manifest_slots = {str(item.get("claim_id")): item for item in (manifest or {}).get("slot_claims", ()) if isinstance(item, dict)}
            manifest_options = {str(item.get("claim_id")): item for item in (manifest or {}).get("elimination_claims", ()) if isinstance(item, dict)}
            manifest_invalid = manifest is None or action.get("candidate_version") != manifest.get("candidate_version") or action.get("manifest_digest") != manifest.get("manifest_digest")
            slot_checks = [dict(item, slot=manifest_slots[str(item.get("claim_id"))].get("slot")) for item in slot_checks if isinstance(item, dict) and str(item.get("claim_id")) in manifest_slots]
            option_checks = [dict(item, option=manifest_options[str(item.get("claim_id"))].get("option")) for item in option_checks if isinstance(item, dict) and str(item.get("claim_id")) in manifest_options]
            if manifest_options:
                candidate_eliminated = {
                    str(item.get("option", "")).strip().upper(): str(item.get("solver_reason", ""))
                    for item in manifest_options.values()
                    if str(item.get("option", "")).strip().upper() != candidate.get("answer", candidate.get("solver_choice", "")).strip().upper()
                }
            expected_slots = {
                int(item["slot"]) for item in candidate_slots
                if isinstance(item, dict) and isinstance(item.get("slot"), int)
            }
            checked_slots = {
                int(item["slot"]) for item in slot_checks
                if isinstance(item, dict) and isinstance(item.get("slot"), int)
            }
            expected_options = {
                str(item.get("option", "")).strip().upper()
                for item in manifest_options.values()
                if str(item.get("option", "")).strip()
            } or {
                str(option).strip().upper() for option in candidate_eliminated
                if str(option).strip()
            }
            checked_options = {
                str(item.get("option", "")).strip().upper()
                for item in option_checks if isinstance(item, dict)
            }
            missing_slots = sorted(expected_slots - checked_slots)
            missing_options = sorted(expected_options - checked_options)
            extra_slots = sorted(checked_slots - expected_slots)
            extra_options = sorted(checked_options - expected_options)
            consistency_errors: list[dict[str, Any]] = []
            assessment_status = {
                "fits": "claim_supported",
                "violates": "claim_contradicted",
                "unknown": "insufficient",
            }
            for item in slot_checks:
                if isinstance(item, dict):
                    item["status"] = assessment_status.get(item.get("assessment"), "insufficient")
                    item["derived_verdict"] = item["status"]
            # The model only supplies an independent semantic assessment.  The
            # kernel derives whether the Solver's claim was supported from the
            # Solver assessment recorded in the immutable manifest.  This
            # removes the rejected+violates / supported double-negative
            # that previously caused every rejected option to be inverted.
            for item in option_checks:
                if not isinstance(item, dict):
                    continue
                assessment = item.get("assessment")
                if assessment == "unknown":
                    status = "insufficient"
                else:
                    manifest_item = manifest_options.get(str(item.get("claim_id")), {})
                    solver_assessment = str(manifest_item.get("solver_assessment", "")).strip().lower()
                    if solver_assessment == "rejected":
                        status = "claim_supported" if assessment == "violates" else "claim_contradicted"
                    elif solver_assessment == "selected":
                        status = "claim_supported" if assessment == "fits" else "claim_contradicted"
                    else:
                        status = "insufficient"
                item["status"] = status
                item["derived_verdict"] = status
            contradicted = [
                item for item in (*slot_checks, *option_checks)
                if isinstance(item, dict) and item.get("status") == "claim_contradicted"
            ]
            insufficient = [
                item for item in (*slot_checks, *option_checks)
                if isinstance(item, dict) and item.get("status") == "insufficient"
            ]
            selected_assessment = selected_check.get("assessment") if isinstance(selected_check, dict) else None
            selected_status = assessment_status.get(selected_assessment, "insufficient")
            # This is a deterministic protocol check, not a semantic judgment:
            # the Critic must not label a claim `fits` while explaining that it
            # fails, or label it `violates` while explaining that it fits.
            def _polarity_error(item: Any, scope: str) -> None:
                if not isinstance(item, dict):
                    return
                assessment = str(item.get("assessment", "")).strip().lower()
                reason_text = str(item.get("reason", "")).casefold()
                negative = any(marker in reason_text for marker in (
                    "contradict", "does not fit", "doesn't fit", "not fit", "wrong", "fails", "incompatible",
                ))
                positive = any(marker in reason_text for marker in (
                    "fits", "satisfies", "satisfied", "correctly completes", "compatible", "supports",
                ))
                if assessment == "fits" and negative and not positive:
                    consistency_errors.append({"scope": scope, "assessment": assessment, "error": "fits_with_negative_reason"})
                elif assessment == "violates" and positive and not negative:
                    consistency_errors.append({"scope": scope, "assessment": assessment, "error": "violates_with_positive_reason"})
            _polarity_error(selected_check, "selected_option_check")
            for item in slot_checks:
                _polarity_error(item, f"slot:{item.get('slot') if isinstance(item, dict) else ''}")
            for item in option_checks:
                _polarity_error(item, f"option:{item.get('option') if isinstance(item, dict) else ''}")
            # The aggregate selected-option assessment cannot claim that the
            # answer fits when one of its required slots is explicitly
            # contradicted.  This compares two Critic outputs only; it does
            # not decide whether the slot is semantically right.
            if selected_assessment == "fits" and any(
                isinstance(item, dict) and item.get("assessment") == "violates"
                for item in slot_checks
            ):
                consistency_errors.append({
                    "scope": "selected_option_check",
                    "assessment": "fits",
                    "error": "selected_fits_but_slot_violates",
                })
            candidate_answer = str(candidate.get("answer", candidate.get("solver_choice", ""))).strip().upper()
            claim_status_mode = any(
                isinstance(item, dict) and item.get("status") in {"claim_supported", "claim_contradicted"}
                for item in (slot_checks, option_checks)
                for item in item
            ) or (
                isinstance(selected_check, dict)
                and selected_check.get("assessment") in {"fits", "violates", "unknown"}
            )
            choice_terms: dict[str, str] = {}
            for node in self.store.snapshot().nodes.values():
                if node.task_id != self.task_id or node.branch_id != self.branch_id or node.type != "choice_source":
                    continue
                raw = node.content
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = {}
                if isinstance(raw, dict):
                    label = str(raw.get("label", "")).strip().upper()
                    if label:
                        choice_terms[label] = str(raw.get("text", ""))
            reason_scope_errors: list[str] = []
            invalid_reason_claims: list[dict[str, Any]] = []
            if claim_status_mode:
                def _mentions_concrete(reason: Any, *values: Any) -> bool:
                    text = str(reason or "").casefold()
                    # Require an exact label/term for short values (e.g. numeric
                    # choices), or at least one distinctive token for prose.
                    for value in values:
                        value_text = str(value or "").strip().casefold()
                        if not value_text:
                            continue
                        if value_text in text:
                            return True
                        tokens = [token for token in re.findall(r"[\w]+", value_text) if len(token) >= 4]
                        if tokens and any(token in text for token in tokens):
                            return True
                    return False

                selected_reason = str(selected_check.get("reason", "")) if isinstance(selected_check, dict) else ""
                selected_text = choice_terms.get(candidate_answer, "")
                # candidate_version + selected_option_check already bind this
                # reason to the selected label.  Requiring the free-form prose
                # to repeat a one-letter label is redundant and produces false
                # positives ("D" occurs inside ordinary words).  Validate only
                # the concrete source content here.
                if candidate_answer and not _mentions_concrete(selected_reason, selected_text):
                    reason_scope_errors.append(
                        "selected_option_check.reason must cite concrete selected option text"
                    )
                    invalid_reason_claims.append({
                        "claim_id": "selected_option_check",
                        "missing_references": ["selected_option_text"],
                        "option": candidate_answer,
                        "option_text": selected_text,
                    })
                for item in slot_checks:
                    if not isinstance(item, dict):
                        continue
                    reason_text = str(item.get("reason", ""))
                    selected_term = next((str(slot.get("selected_term", "")) for slot in candidate_slots if isinstance(slot, dict) and slot.get("slot") == item.get("slot")), "")
                    # claim_id is the authoritative slot identity.  "first
                    # blank" and "slot 1" are equivalent prose, so do not
                    # require the numeral to be repeated in reason.
                    if not _mentions_concrete(reason_text, selected_term):
                        reason_scope_errors.append(f"slot {item.get('slot')} reason must cite its selected term")
                        invalid_reason_claims.append({
                            "claim_id": str(item.get("claim_id", "")),
                            "missing_references": ["selected_term"],
                            "slot": item.get("slot"),
                            "selected_term": selected_term,
                        })
                for item in option_checks:
                    if not isinstance(item, dict):
                        continue
                    option = str(item.get("option", "")).strip().upper()
                    reason_text = str(item.get("reason", ""))
                    # The manifest claim_id already binds the option label.
                    # Require concrete source text, not a duplicated A/B/C
                    # marker in natural-language prose.
                    if not _mentions_concrete(reason_text, choice_terms.get(option, "")):
                        reason_scope_errors.append(f"option {option} reason must cite distinctive option text")
                        invalid_reason_claims.append({
                            "claim_id": str(item.get("claim_id", "")),
                            "missing_references": ["option_text"],
                            "option": option,
                            "option_text": choice_terms.get(option, ""),
                        })
            if manifest_invalid:
                status = "uncertain"
                error_type = "invalid_choice_audit_manifest"
                error_location = "manifest"
                reason = "Critic audit manifest does not match the current candidate version."
                repair_instruction = "Re-read the immutable Audit Manifest and emit its exact candidate_version and manifest_digest."
            elif consistency_errors:
                status = "uncertain"
                error_type = "invalid_choice_audit_consistency"
                error_location = "audit_fields"
                reason = json.dumps({
                    "reason": "Solver assessment, Critic assessment, and verdict are inconsistent.",
                    "repair_packet": {
                        "status": "critic_consistency_repair",
                        "errors": consistency_errors,
                        "required": "Preserve claim IDs and correct only verdict or reason.",
                    },
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Fix only the deterministic field consistency errors; do not change Solver assessments or claim IDs."
            elif reason_scope_errors:
                status = "uncertain"
                error_type = "invalid_choice_audit_reason"
                error_location = "audit_reason"
                reason = json.dumps({
                    "reason": "; ".join(reason_scope_errors),
                    "repair_packet": {
                        "status": "critic_reason_repair",
                        "error_type": "invalid_choice_audit_reason",
                        "invalid_claims": invalid_reason_claims,
                        "required": "Rewrite only the listed reasons and cite every required concrete value.",
                        "preserve": ["candidate_version", "manifest_digest", "claim_id"],
                    },
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Rewrite only the listed Critic reasons and cite the concrete selected term or option text; claim_id already identifies its slot or label."
            contradicted_slots = [
                {
                    "slot": item.get("slot"),
                    "previous_term": next(
                        (slot.get("selected_term", "") for slot in candidate_slots
                         if isinstance(slot, dict) and slot.get("slot") == item.get("slot")),
                        "",
                    ),
                    "reason": str(item.get("reason", "")),
                }
                for item in slot_checks
                if isinstance(item, dict) and item.get("status") in {"contradicted", "claim_contradicted"}
            ]
            contradicted_options = [
                {
                    "option": str(item.get("option", "")).strip().upper(),
                    "previous_reason": str(candidate_eliminated.get(item.get("option"), candidate_eliminated.get(str(item.get("option", "")).upper(), ""))),
                    "reason": str(item.get("reason", "")),
                }
                for item in option_checks
                if isinstance(item, dict) and item.get("status") in {"contradicted", "claim_contradicted"}
            ]
            fitting_options = [
                str(item.get("option", "")).strip().upper()
                for item in option_checks
                if isinstance(item, dict)
                and item.get("assessment") == "fits"
                and str(item.get("option", "")).strip().upper() != candidate_answer
            ]
            repair_packet = {
                "candidate_answer": candidate_answer,
                "selected_option_check": selected_check,
                "contradicted_slots": contradicted_slots,
                "contradicted_eliminations": contradicted_options,
                # Any contradicted Solver claim invalidates the candidate as a
                # whole.  The repair contract therefore requires a genuinely
                # new choice; otherwise the model can cosmetically rewrite a
                # reason while preserving the rejected answer.
                "required_change": "selected_answer" if (selected_status == "claim_contradicted" or contradicted_slots or contradicted_options) else "audited_claims",
                "forbidden_answers": [candidate_answer] if (selected_status == "claim_contradicted" or contradicted_slots or contradicted_options) and candidate_answer else [],
                # These are options the Critic independently found to fit
                # with the source.  They are candidates for Solver repair;
                # they are deliberately not inferred from claim polarity.
                "viable_challengers": list(dict.fromkeys(fitting_options)),
                "selected_claim_conflicts_with_slots": bool(selected_status == "claim_supported" and contradicted_slots),
            }
            # Preserve prior rejected choices across repair rounds so the
            # Solver cannot oscillate A -> D -> A when each round only bans the
            # immediately previous candidate.
            historical_rejections: list[str] = []
            for node in self.store.snapshot().nodes.values():
                if node.task_id != self.task_id or node.branch_id != self.branch_id or node.logical_id != "verification":
                    continue
                if node.status not in {"need_fix", "uncertain"} or not isinstance(node.content, dict):
                    continue
                old_reason = node.content.get("reason", "")
                if not isinstance(old_reason, str) or not old_reason.strip().startswith("{"):
                    continue
                try:
                    old_packet = json.loads(old_reason).get("repair_packet", {})
                except json.JSONDecodeError:
                    old_packet = {}
                if isinstance(old_packet, dict):
                    historical_rejections.extend(str(item).strip().upper() for item in old_packet.get("forbidden_answers", ()) if str(item).strip())
            repair_packet["audit_history"] = {"rejected_answers": list(dict.fromkeys(historical_rejections))}
            repair_packet["forbidden_answers"] = list(dict.fromkeys([*historical_rejections, *repair_packet["forbidden_answers"]]))
            if consistency_errors:
                status = "uncertain"
                error_type = "invalid_choice_audit_consistency"
                error_location = "audit_fields"
                reason = json.dumps({
                    "reason": "Solver assessment, Critic assessment, and verdict are inconsistent.",
                    "repair_packet": {"status": "critic_consistency_repair", "errors": consistency_errors, "required": "Preserve claim IDs; correct only verdict or reason."},
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Fix only the deterministic field consistency errors; preserve claim IDs."
            elif reason_scope_errors:
                status = "uncertain"
                error_type = "invalid_choice_audit_reason"
                error_location = "audit_reason"
                reason = json.dumps({
                    "reason": "; ".join(reason_scope_errors),
                    "repair_packet": {
                        "status": "critic_reason_repair",
                        "error_type": "invalid_choice_audit_reason",
                        "invalid_claims": invalid_reason_claims,
                        "required": "Rewrite only the listed reasons and cite every required concrete value.",
                        "preserve": ["candidate_version", "manifest_digest", "claim_id"],
                    },
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Rewrite only the listed Critic reasons and cite the concrete selected term or option text; claim_id already identifies its slot or label."
            elif selected_status == "claim_contradicted":
                status = "need_fix"
                error_type = "contradicted_selected_option"
                error_location = "selected_option"
                reason = json.dumps({
                    "reason": str(selected_check.get("reason", "The selected option was contradicted.")),
                    "repair_packet": repair_packet,
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Re-evaluate the selected option against the original question and choose a corrected option."
            elif selected_status == "insufficient":
                status = "uncertain"
                error_type = "insufficient_selected_option_audit"
                error_location = "selected_option"
                reason = json.dumps({
                    "reason": str(selected_check.get("reason", "The selected option was not sufficiently audited.")),
                    "repair_packet": repair_packet,
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Re-audit the complete source. Rewrite the selected-option and slot reasons with concrete question constraints; do not copy the previous candidate unchanged."
            elif missing_slots or missing_options or extra_slots or extra_options:
                status = "uncertain"
                error_type = "invalid_choice_audit_scope"
                error_location = "slot_checks" if missing_slots or extra_slots else "option_checks"
                reason = (
                    f"Audit coverage must exactly match Solver claims; missing slots={missing_slots}, "
                    f"extra slots={extra_slots}, missing options={missing_options}, extra options={extra_options}."
                )
                repair_instruction = "Audit exactly every Solver slot and every eliminated option, with no extra claims."
            elif contradicted:
                status = "need_fix"
                error_type = "contradicted_choice_claim"
                error_location = "slot_analysis" if any(item in slot_checks for item in contradicted) else "option_reviews"
                reason = json.dumps({
                    "reason": action["summary"],
                    "repair_packet": repair_packet,
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Revise the Solver choice and the contradicted analysis claims."
            elif insufficient:
                status = "uncertain"
                error_type = "insufficient_choice_audit"
                error_location = "slot_analysis"
                reason = json.dumps({
                    "reason": action["summary"],
                    "repair_packet": repair_packet,
                }, ensure_ascii=False, separators=(",", ":"))
                repair_instruction = "Provide concrete checks for every Solver claim before verification; rewrite the candidate analysis instead of copying it unchanged."
            else:
                status = "verified"
                error_type = ""
                error_location = ""
                reason = action["summary"]
                repair_instruction = ""
        elif op == "diagnose_code":
            execution = self._latest_node("execution") or self._latest_node("result")
            execution_valid = bool(execution and execution.validation.get("execution_valid"))
            status = "verified" if execution_valid else "need_fix"
            error_type = "" if execution_valid else "execution_failure"
            error_location = "" if execution_valid else "code"
            reason = action["diagnosis"]
            repair_instruction = "" if execution_valid else action["repair_instruction"]
        elif op == "judge_evidence":
            supported = (
                action["answer_support"] == "supported"
                and action["bridge_status"] == "valid"
                and action["citation_status"] == "valid"
                and action["repair_target"] == "none"
            )
            uncertain = "uncertain" in {action["answer_support"], action["bridge_status"]} or action["citation_status"] == "insufficient"
            status = "verified" if supported else "uncertain" if uncertain else "need_fix"
            error_type = "" if supported else (
                action["citation_status"] if action["citation_status"] != "valid" else action["answer_support"]
            )
            error_location = "" if supported else action["repair_target"]
            reason = action["reason"]
            repair_instruction = "" if supported else f"Repair {action['repair_target']} using visible evidence."
        return {
            "op": "verify",
            "target": target,
            "status": status,
            "error_type": error_type,
            "error_location": error_location,
            "reason": reason,
            "repair_instruction": repair_instruction,
            "preserve": [],
            "requested_fragments": [],
        }

    def copy_verified_result(self) -> CompilationResult:
        verification = self._latest_node("verification")
        if verification is None or verification.status != "verified":
            raise ActionCompilationError("deterministic finalizer requires a verified verification node")
        state = self.store.snapshot()
        targets = [
            state.nodes.get(edge.target)
            for edge in state.edges
            if edge.source == verification.node_id and edge.relation == "verifies"
        ]
        targets = [node for node in targets if node is not None and node.is_operationally_valid()]
        if not targets:
            raise ActionCompilationError("verified verification node has no valid verifies target")
        target = max(targets, key=lambda node: (node.version, node.created_at, node.node_id))
        content = target.content
        if self.task_type == "code_generation":
            value = content.get("code", content.get("value", content)) if isinstance(content, dict) else content
            if isinstance(value, dict):
                value = value.get("code", value)
        else:
            value = content.get("value", content) if isinstance(content, dict) else content
        if self._same_latest_content("final_answer", value):
            node_ids: tuple[str, ...] = ()
        else:
            final_id = self._add_node(
                "runtime",
                logical_id="final_answer",
                node_type="final_answer",
                content=value,
                status="verified",
                provenance={"source_node_id": target.node_id, "kernel_finalized": True},
            )
            self.store.add_edge(source=target.node_id, target=final_id, relation="derived_from", created_by_role="runtime")
            node_ids = (final_id,)
        return CompilationResult(
            action={"op": "copy_verified_result", "source": target.logical_id},
            node_ids=node_ids,
        )

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

    def _latest_node(self, logical_id: str):
        return self.store.latest_valid(self.task_id, self.branch_id, logical_id)

    def _same_latest_content(self, logical_id: str, content: Any) -> bool:
        node = self._latest_node(logical_id)
        return node is not None and node.content == content

    def _resolve_target(self, ref: Any):
        logical_id = str(ref.get("logical_id", "") if isinstance(ref, dict) else ref).strip()
        state = self.store.snapshot()
        if logical_id in state.nodes:
            return state.nodes[logical_id]
        for index_name in ("fragment_index", "local_node_index", "semantic_type_index"):
            index = self._node_ref_context.get(index_name, {})
            value = index.get(logical_id) if isinstance(index, dict) else None
            if isinstance(value, str):
                node = state.nodes.get(value) or self._latest_node(value)
                if node is not None:
                    return node
            if isinstance(value, dict):
                node_id = str(value.get("source_node_id", value.get("node_id", "")))
                if node_id in state.nodes:
                    return state.nodes[node_id]
                source_logical_id = str(value.get("source_logical_id", value.get("logical_id", "")))
                if source_logical_id:
                    node = self._latest_node(source_logical_id)
                    if node is not None:
                        return node
        return self._latest_node(logical_id)

    def _compile_repair_patch(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        current = self._latest_node("result")
        if current is None:
            raise ActionCompilationError("repair_patch requires an existing result")
        content = current.content
        value = content.get("value") if isinstance(content, dict) and "value" in content else content
        if not isinstance(value, dict):
            raise ActionCompilationError("repair_patch requires a structured result value")
        merged = copy.deepcopy(value)
        patch = action["patch"]
        action_name = action["action"]
        if action_name in {"ADD_EVIDENCE", "REPLACE_EVIDENCE", "PATCH_CITATIONS"}:
            selected = [str(item).strip() for item in patch["selected_fact_ids"]]
            if action_name == "ADD_EVIDENCE":
                merged["cited_fact_ids"] = list(dict.fromkeys([*merged.get("cited_fact_ids", []), *selected]))
            else:
                merged["cited_fact_ids"] = selected
        elif action_name == "PATCH_BRIDGE":
            merged["bridge_entity"] = patch["bridge_entity"].strip()
        elif action_name == "REVISE_ANSWER":
            merged["answer"] = patch["answer"].strip()
        else:
            raise ActionCompilationError(f"unsupported repair_patch action: {action_name}")
        merged["reasoning_chain"] = copy.deepcopy(patch["reasoning_chain"])
        result = {"id": "answer", "value": merged}
        if self._same_latest_content("result", result):
            return ()
        result_id = self._add_node(role, logical_id="result", node_type="result", content=result)
        self.store.add_edge(source=current.node_id, target=result_id, relation="derived_from", created_by_role=role)
        yield result_id

    def _compile_kernel_verification(self, role: str, action: dict[str, Any]) -> Iterable[str]:
        target = self._resolve_target(action["target"])
        if target is None:
            raise ActionCompilationError(f"verification target does not exist: {action['target']}")
        verified = action["status"] == "verified"
        content = {"target": target.logical_id, "status": action["status"]}
        if action.get("reason"):
            content["reason"] = action["reason"]
        if not verified:
            content.update({
                "error_type": action["error_type"],
                "error_location": action["error_location"],
                "repair_instruction": action["repair_instruction"],
                "preserve": list(action.get("preserve", [])),
                "requested_fragments": list(action.get("requested_fragments", [])),
            })
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
