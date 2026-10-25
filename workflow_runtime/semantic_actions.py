"""Small model-facing semantic actions lowered to the internal Delta IR."""

from __future__ import annotations

import copy
from typing import Any, Mapping


PLANNER_OP = "plan"
SOLVER_OP_BY_TASK = {
    "numeric_solve": "solve_numeric",
    "numeric_comparison": "solve_numeric",
    "multiple_choice": "choose_option",
    "code_generation": "submit_code",
    "multihop_qa": "answer_with_evidence",
}
CRITIC_OP_BY_TASK = {
    "numeric_solve": "judge_numeric",
    "numeric_comparison": "judge_numeric",
    "multiple_choice": "audit_choice",
    "code_generation": "diagnose_code",
    "multihop_qa": "judge_evidence",
}
TYPED_SEMANTIC_OPS = frozenset({PLANNER_OP, *SOLVER_OP_BY_TASK.values()})
CRITIC_TYPED_OPS = frozenset(CRITIC_OP_BY_TASK.values())
MODEL_TYPED_OPS = frozenset({*TYPED_SEMANTIC_OPS, *CRITIC_TYPED_OPS})


def model_semantic_ops(role: str, task_type: str, *, repair: bool = False) -> tuple[str, ...]:
    if role == "planner":
        return (PLANNER_OP,)
    if role == "solver":
        if repair and task_type == "multihop_qa":
            return ("repair_patch",)
        op = SOLVER_OP_BY_TASK.get(task_type)
        return (op,) if op else ()
    if role == "critic":
        op = CRITIC_OP_BY_TASK.get(task_type)
        return (op,) if op else ()
    return ()


def semantic_action_example(role: str, task_type: str) -> dict[str, Any]:
    if role == "planner":
        facts = (
            [{"content": "the task states a quantity of 2", "source": "task", "quote": ""}]
            if task_type in {"numeric_solve", "numeric_comparison"}
            else []
        )
        strategy = {
            "multiple_choice": "compare every option and select the best supported choice",
            "code_generation": "implement the requested function and check its behavior",
            "multihop_qa": "connect two evidence hops and answer from cited sentences",
        }.get(task_type, "derive the requested numeric result from the stated quantities")
        return {"op": "plan", "facts": facts, "strategy": strategy}
    if role == "critic":
        if task_type in {"numeric_solve", "numeric_comparison"}:
            return {
                "op": "judge_numeric", "model_verdict": "consistent",
                "error_type": "none", "error_location": "", "reason": "consistent",
                "repair_instruction": "",
            }
        if task_type == "multiple_choice":
            return {
                "op": "audit_choice",
                "candidate_version": "result@v1",
                "manifest_digest": "manifest-digest",
                "selected_option_check": {"assessment": "unknown", "reason": "<cite the exact selected option text and question constraint>"},
                "slot_checks": [{
                    "claim_id": "result@v1:slot:1",
                    "assessment": "unknown",
                    "reason": "<cite its exact selected term and the matching question constraint>",
                }],
                "option_checks": [{
                    "claim_id": "result@v1:elimination:B",
                    "assessment": "unknown",
                    "reason": "<cite the option text and the exact reason it conflicts with the question>",
                }],
                "summary": "<summarize the independently audited fits, violates, and unknown claims>",
            }
        if task_type == "code_generation":
            return {
                "op": "diagnose_code", "diagnosis": "The test signal is consistent.",
                "repair_instruction": "",
            }
        if task_type == "multihop_qa":
            return {
                "op": "judge_evidence", "answer_support": "supported",
                "bridge_status": "valid", "citation_status": "valid",
                "repair_target": "none", "reason": "Both evidence hops support the answer.",
            }
    if task_type in {"numeric_solve", "numeric_comparison"}:
        return {"op": "solve_numeric", "expression": "2+3", "value": "5"}
    if task_type == "multiple_choice":
        return {
            "op": "choose_option",
                "answer": "A",
                "slot_reasons": [{"slot": 1, "reason": "the selected term fits the question"}],
            "option_reviews": [
                {"label": "A", "assessment": "selected", "reason": "the selected option fits"},
                {"label": "B", "assessment": "rejected", "reason": "contradicts the question"},
            ],
            "confidence": 0.5,
        }
    if task_type == "code_generation":
        return {"op": "submit_code", "code": "def solution():\n    pass"}
    if task_type == "multihop_qa":
        return {
            "op": "answer_with_evidence",
            "answer": "...",
            "bridge_entity": "...",
            "citations": ["sentence_source_1", "sentence_source_2"],
            "reasoning": ["first evidence hop", "second evidence hop"],
        }
    return {"op": "done"}


def validate_typed_semantic_action(role: str, task_type: str, payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["typed semantic action must be an object"]
    op = str(payload.get("op", ""))
    expected_ops = model_semantic_ops(role, task_type)
    if op not in expected_ops:
        return [f"{role} task {task_type} cannot emit typed semantic action {op}"]

    errors: list[str] = []
    if op == "plan":
        expected = {"op", "facts", "strategy"}
        if set(payload) != expected:
            errors.append("plan keys must be exactly facts, op, strategy")
        facts = payload.get("facts")
        if not isinstance(facts, list):
            errors.append("plan.facts must be an array")
            facts = []
        if task_type in {"numeric_solve", "numeric_comparison"} and not facts:
            errors.append("numeric plan requires at least one grounded fact")
        if task_type not in {"numeric_solve", "numeric_comparison"} and facts:
            errors.append("non-numeric plan must use facts=[] and reference Source Graph through its strategy")
        if not isinstance(payload.get("strategy"), str) or not str(payload.get("strategy", "")).strip():
            errors.append("plan.strategy must be a non-empty string")
        for index, fact in enumerate(facts):
            if not isinstance(fact, Mapping) or set(fact) != {"content", "source", "quote"}:
                errors.append(f"plan.facts[{index}] keys must be exactly content, quote, source")
                continue
            if not isinstance(fact.get("content"), str) or not str(fact.get("content", "")).strip():
                errors.append(f"plan.facts[{index}].content must be a non-empty string")
            if not isinstance(fact.get("source"), str) or not str(fact.get("source", "")).strip():
                errors.append(f"plan.facts[{index}].source must be a non-empty string")
            if not isinstance(fact.get("quote"), str):
                errors.append(f"plan.facts[{index}].quote must be a string")
    elif op == "solve_numeric":
        if set(payload) != {"op", "expression", "value"}:
            errors.append("solve_numeric keys must be exactly expression, op, value")
        if not isinstance(payload.get("expression"), str) or not str(payload.get("expression", "")).strip():
            errors.append("solve_numeric.expression must be a non-empty string")
        if not isinstance(payload.get("value"), str) or not str(payload.get("value", "")).strip():
            errors.append("solve_numeric.value must be a non-empty string")
    elif op == "choose_option":
        expected = {"op", "answer", "slot_reasons", "option_reviews", "confidence"}
        if set(payload) != expected:
            errors.append("choose_option keys must be exactly answer, confidence, op, option_reviews, slot_reasons")
        if not isinstance(payload.get("answer"), str) or not str(payload.get("answer", "")).strip():
            errors.append("choose_option.answer must be a non-empty string")
        slots = payload.get("slot_reasons")
        if not isinstance(slots, list) or not slots:
            errors.append("choose_option.slot_reasons must be a non-empty array")
        else:
            seen_slots: set[int] = set()
            for index, item in enumerate(slots):
                if not isinstance(item, Mapping) or set(item) != {"slot", "reason"}:
                    errors.append(f"choose_option.slot_reasons[{index}] must contain exactly slot, reason")
                    continue
                slot = item.get("slot")
                if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
                    errors.append(f"choose_option.slot_reasons[{index}].slot must be a positive integer")
                elif slot in seen_slots:
                    errors.append(f"choose_option.slot_reasons contains duplicate slot {slot}")
                else:
                    seen_slots.add(slot)
                for key in ("reason",):
                    if not isinstance(item.get(key), str) or not str(item.get(key, "")).strip():
                        errors.append(f"choose_option.slot_reasons[{index}].{key} must be a non-empty string")
        reviews = payload.get("option_reviews")
        if not isinstance(reviews, list) or not reviews:
            errors.append("choose_option.option_reviews must be a non-empty array")
        else:
            answer = str(payload.get("answer", "")).strip().upper()
            seen_labels: set[str] = set()
            for index, item in enumerate(reviews):
                if not isinstance(item, Mapping) or set(item) != {"label", "assessment", "reason"}:
                    errors.append(f"choose_option.option_reviews[{index}] must contain exactly label, assessment, reason")
                    continue
                label = str(item.get("label", "")).strip().upper()
                if not label:
                    errors.append(f"choose_option.option_reviews[{index}].label must be non-empty")
                elif label in seen_labels:
                    errors.append(f"choose_option.option_reviews contains duplicate label {label}")
                seen_labels.add(label)
                if item.get("assessment") not in {"selected", "rejected"}:
                    errors.append(f"choose_option.option_reviews[{index}].assessment must be selected or rejected")
                if not isinstance(item.get("reason"), str) or not str(item.get("reason", "")).strip():
                    errors.append(f"choose_option.option_reviews[{index}].reason must be a non-empty string")
            selected = [str(item.get("label", "")).strip().upper() for item in reviews if isinstance(item, Mapping) and item.get("assessment") == "selected"]
            if selected != [answer]:
                errors.append("choose_option.option_reviews must contain exactly one selected label equal to answer")
        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or confidence not in {0, 0.25, 0.5, 0.75, 0.9, 1}:
            errors.append("choose_option.confidence must be one of 0, 0.25, 0.5, 0.75, 0.9, 1")
    elif op == "submit_code":
        if set(payload) != {"op", "code"}:
            errors.append("submit_code keys must be exactly code, op")
        if not isinstance(payload.get("code"), str) or not str(payload.get("code", "")).strip():
            errors.append("submit_code.code must be a non-empty string")
    elif op == "answer_with_evidence":
        expected = {"op", "answer", "bridge_entity", "citations", "reasoning"}
        if set(payload) != expected:
            errors.append("answer_with_evidence keys must be exactly answer, bridge_entity, citations, op, reasoning")
        for key in ("answer", "bridge_entity"):
            if not isinstance(payload.get(key), str) or not str(payload.get(key, "")).strip():
                errors.append(f"answer_with_evidence.{key} must be a non-empty string")
        for key in ("citations", "reasoning"):
            value = payload.get(key)
            if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
                errors.append(f"answer_with_evidence.{key} must be a non-empty array of strings")
        if isinstance(payload.get("citations"), list) and len(payload["citations"]) < 2:
            errors.append("answer_with_evidence requires at least two citations")
    elif op == "judge_numeric":
        expected = {"op", "model_verdict", "error_type", "error_location", "reason", "repair_instruction"}
        if set(payload) != expected:
            errors.append("judge_numeric keys must be exactly error_location, error_type, model_verdict, op, reason, repair_instruction")
        verdict = payload.get("model_verdict")
        if verdict not in {"consistent", "inconsistent", "uncertain"}:
            errors.append("judge_numeric.model_verdict must be consistent, inconsistent, or uncertain")
        if payload.get("error_type") not in {"none", "arithmetic", "wrong_operation", "missing_operand", "unit_mismatch", "insufficient_context"}:
            errors.append("judge_numeric.error_type is invalid")
        if payload.get("error_location") not in {"", "expression", "value", "grounding"}:
            errors.append("judge_numeric.error_location is invalid")
        for key in ("reason", "repair_instruction"):
            if not isinstance(payload.get(key), str):
                errors.append(f"judge_numeric.{key} must be a string")
        if verdict == "consistent" and (
            payload.get("error_type") != "none"
            or payload.get("error_location") != ""
            or payload.get("repair_instruction") != ""
        ):
            errors.append("consistent judge_numeric must have no error, location, or repair instruction")
        if verdict == "inconsistent" and (
            payload.get("error_type") in {None, "none", "insufficient_context"}
            or not payload.get("error_location")
            or not str(payload.get("reason", "")).strip()
            or not str(payload.get("repair_instruction", "")).strip()
        ):
            errors.append("inconsistent judge_numeric requires a concrete error, location, reason, and repair instruction")
    elif op == "audit_choice":
        expected = {"op", "candidate_version", "manifest_digest", "selected_option_check", "slot_checks", "option_checks", "summary"}
        if set(payload) != expected:
            errors.append("audit_choice keys must be exactly candidate_version, manifest_digest, op, option_checks, selected_option_check, slot_checks, summary")
        if not isinstance(payload.get("candidate_version"), str) or not str(payload.get("candidate_version", "")).strip():
            errors.append("audit_choice.candidate_version must be non-empty")
        if not isinstance(payload.get("manifest_digest"), str) or not str(payload.get("manifest_digest", "")).strip():
            errors.append("audit_choice.manifest_digest must be non-empty")
        selected = payload.get("selected_option_check")
        if not isinstance(selected, Mapping) or set(selected) != {"assessment", "reason"}:
            errors.append("audit_choice.selected_option_check must contain exactly assessment, reason")
        else:
            if selected.get("assessment") not in {"fits", "violates", "unknown"}:
                errors.append("audit_choice.selected_option_check.assessment is invalid")
            if not isinstance(selected.get("reason"), str) or not selected["reason"].strip():
                errors.append("audit_choice.selected_option_check.reason must be non-empty")
        slots = payload.get("slot_checks")
        if not isinstance(slots, list) or not slots:
            errors.append("audit_choice.slot_checks must be a non-empty array")
        else:
            seen: set[str] = set()
            for index, item in enumerate(slots):
                if not isinstance(item, Mapping) or set(item) != {"claim_id", "assessment", "reason"}:
                    errors.append(f"audit_choice.slot_checks[{index}] must contain exactly claim_id, assessment, reason")
                    continue
                claim_id = str(item.get("claim_id", "")).strip()
                if not claim_id or claim_id in seen:
                    errors.append(f"audit_choice.slot_checks[{index}].claim_id must be unique and non-empty")
                else:
                    seen.add(claim_id)
                if item.get("assessment") not in {"fits", "violates", "unknown"}:
                    errors.append(f"audit_choice.slot_checks[{index}].assessment is invalid")
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    errors.append(f"audit_choice.slot_checks[{index}].reason must be non-empty")
        checks = payload.get("option_checks")
        if not isinstance(checks, list) or not checks:
            errors.append("audit_choice.option_checks must be a non-empty array")
        else:
            seen_options: set[str] = set()
            for index, item in enumerate(checks):
                if not isinstance(item, Mapping) or set(item) != {"claim_id", "assessment", "reason"}:
                    errors.append(f"audit_choice.option_checks[{index}] must contain exactly claim_id, assessment, reason")
                    continue
                claim_id = str(item.get("claim_id", "")).strip()
                if not claim_id or claim_id in seen_options:
                    errors.append(f"audit_choice.option_checks[{index}].claim_id must be unique and non-empty")
                else:
                    seen_options.add(claim_id)
                if item.get("assessment") not in {"fits", "violates", "unknown"}:
                    errors.append(f"audit_choice.option_checks[{index}].assessment is invalid")
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    errors.append(f"audit_choice.option_checks[{index}].reason must be non-empty")
        if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
            errors.append("audit_choice.summary must be non-empty")
    elif op == "diagnose_code":
        expected = {"op", "diagnosis", "repair_instruction"}
        if set(payload) != expected:
            errors.append("diagnose_code keys must be exactly diagnosis, op, repair_instruction")
        if not isinstance(payload.get("diagnosis"), str) or not str(payload.get("diagnosis", "")).strip():
            errors.append("diagnose_code.diagnosis must be a non-empty string")
        if not isinstance(payload.get("repair_instruction"), str):
            errors.append("diagnose_code.repair_instruction must be a string")
    elif op == "judge_evidence":
        expected = {"op", "answer_support", "bridge_status", "citation_status", "repair_target", "reason"}
        if set(payload) != expected:
            errors.append("judge_evidence keys must be exactly answer_support, bridge_status, citation_status, op, reason, repair_target")
        if payload.get("answer_support") not in {"supported", "unsupported", "contradicted", "uncertain"}:
            errors.append("judge_evidence.answer_support is invalid")
        if payload.get("bridge_status") not in {"valid", "invalid", "missing", "uncertain"}:
            errors.append("judge_evidence.bridge_status is invalid")
        if payload.get("citation_status") not in {"valid", "invalid_citation", "missing_hop", "insufficient"}:
            errors.append("judge_evidence.citation_status is invalid")
        if payload.get("repair_target") not in {"none", "answer", "bridge_entity", "cited_fact_ids"}:
            errors.append("judge_evidence.repair_target is invalid")
        if not isinstance(payload.get("reason"), str) or not str(payload.get("reason", "")).strip():
            errors.append("judge_evidence.reason must be a non-empty string")
        fully_supported = (
            payload.get("answer_support") == "supported"
            and payload.get("bridge_status") == "valid"
            and payload.get("citation_status") == "valid"
        )
        if fully_supported != (payload.get("repair_target") == "none"):
            errors.append("judge_evidence repair_target must be none exactly when answer, bridge, and citations are valid")
    return errors


def lower_typed_semantic_action(role: str, task_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_typed_semantic_action(role, task_type, payload)
    if errors:
        raise ValueError("; ".join(errors))
    action = copy.deepcopy(dict(payload))
    op = action["op"]
    if op == "plan":
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        for index, fact in enumerate(action["facts"], start=1):
            temp_id = f"f{index}"
            nodes.append({
                "temp_id": temp_id,
                "type": "fact",
                "content": fact["content"],
                "grounding": [{"source": {"logical_id": fact["source"]}, "quote": fact["quote"]}],
            })
        plan_id = "p1"
        nodes.append({
            "temp_id": plan_id,
            "type": "plan_step",
            "content": {"operation": action["strategy"]},
            "grounding": [],
        })
        edges.extend(
            {"source": f"f{index}", "relation": "input_to", "target": plan_id}
            for index in range(1, len(action["facts"]) + 1)
        )
        return {
            "nodes": nodes,
            "edges": edges,
            "send_hints": [{"receiver": "solver", "fragments": [node["temp_id"] for node in nodes], "reason": "planner output"}],
            "complete": True,
        }
    if op == "solve_numeric":
        return {
            "nodes": [
                {"temp_id": "c1", "type": "calculation", "content": {"expression": action["expression"], "value": action["value"]}, "grounding": []},
                {"temp_id": "a1", "type": "candidate_answer", "content": {"value": action["value"]}, "grounding": []},
            ],
            "edges": [{"source": "c1", "relation": "produces", "target": "a1"}],
            "send_hints": [{"receiver": "critic", "fragments": ["c1", "a1"], "reason": "verify numeric result"}],
            "complete": True,
        }
    if op == "choose_option":
        # selected_term values are source-derived by the kernel after the
        # model selects an option; the model submits only per-slot reasons.
        slot_analysis = [
            {"slot": item["slot"], "selected_term": "", "reason": item["reason"]}
            for item in action["slot_reasons"]
        ]
        content = {
            "answer": action["answer"],
            "solver_choice": action["answer"],
            "slot_analysis": slot_analysis,
            "slot_reasons": action["slot_reasons"],
            "option_reviews": action["option_reviews"],
            "eliminated_options": {
                str(item["label"]).strip().upper(): item["reason"]
                for item in action["option_reviews"]
                if item.get("assessment") == "rejected"
            },
            "confidence": action["confidence"],
        }
        return {
            "nodes": [{"temp_id": "a1", "type": "candidate_answer", "content": content, "grounding": []}],
            "edges": [],
            "send_hints": [{"receiver": "critic", "fragments": ["a1"], "reason": "verify option choice"}],
            "complete": True,
        }
    if op == "submit_code":
        return {
            "nodes": [{"temp_id": "code1", "type": "code", "content": {"code": action["code"]}, "grounding": []}],
            "edges": [],
            "send_hints": [{"receiver": "critic", "fragments": ["code1"], "reason": "verify code"}],
            "complete": True,
        }
    if op == "answer_with_evidence":
        citations = list(action["citations"])
        evidence = {
            "temp_id": "e1",
            "type": "selected_evidence",
            "content": {"sentence_refs": citations, "bridge_entity": action["bridge_entity"]},
            "grounding": [
                {"source": {"logical_id": citation}, "quote": ""}
                for citation in citations
            ],
        }
        answer = {
            "temp_id": "a1",
            "type": "candidate_answer",
            "content": {
                "answer": action["answer"],
                "bridge_entity": action["bridge_entity"],
                "cited_fact_ids": citations,
                "reasoning_chain": list(action["reasoning"]),
            },
            "grounding": [],
        }
        return {
            "nodes": [evidence, answer],
            "edges": [{"source": "e1", "relation": "derived_from", "target": "a1"}],
            "send_hints": [{"receiver": "critic", "fragments": ["e1", "a1"], "reason": "verify evidence answer"}],
            "complete": True,
        }
    raise ValueError(f"unsupported typed semantic action: {op}")
