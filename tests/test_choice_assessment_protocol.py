from workflow_runtime import GraphStore
from workflow_runtime.action_compiler import ActionCompiler
from workflow_runtime.choice_audit import build_choice_audit_manifest
from workflow_runtime.protocol import validate_action


def _store(answer: str = "I"):
    store = GraphStore()
    common = {"task_id": "assessment", "branch_id": "main", "owner": "dataset", "created_by_role": "dataset"}
    store.add_node(logical_id="task", node_type="task_source", content="Fill all four blanks.", **common)
    options = {
        "A": "Safe practices, Fear, Jealousy, Trivial",
        "B": "Unsafe practices, Distress, Joy, Trivial",
        "I": "Unsafe practices, Distress, Fear, Serious",
    }
    for label, text in options.items():
        store.add_node(logical_id=f"choice_source_{label}", node_type="choice_source", content={"label": label, "text": text}, **common)
    compiler = ActionCompiler(store, task_id="assessment", task_type="multiple_choice")
    reviews = [
        {"label": label, "assessment": "selected" if label == answer else "rejected", "reason": text}
        for label, text in options.items()
    ]
    compiler.apply("solver", {
        "op": "choose_option",
        "answer": answer,
        "slot_reasons": [{"slot": i, "reason": term} for i, term in enumerate(answer == "I" and ("Unsafe practices", "Distress", "Fear", "Serious") or ("Safe practices", "Fear", "Jealousy", "Trivial"), 1)],
        "option_reviews": reviews,
        "confidence": 1,
    })
    return store, compiler, build_choice_audit_manifest(store, task_id="assessment")


def test_assessment_protocol_derives_supported_rejected_claims():
    store, compiler, manifest = _store("I")
    action = {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"],
        "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [{"claim_id": item["claim_id"], "assessment": "fits", "reason": item["selected_term"]} for item in manifest["slot_claims"]],
        "option_checks": [{"claim_id": item["claim_id"], "assessment": "fits" if item["option"] == "I" else "violates", "reason": item["option_text"]} for item in manifest["elimination_claims"]],
        "selected_option_check": {"assessment": "fits", "reason": manifest["selected_option_text"]},
        "summary": "all claims audited",
    }
    result = compiler.critic_verification_action(action)
    assert result["status"] == "verified"


def test_assessment_protocol_routes_selected_failure_to_compatible_challenger():
    store, compiler, manifest = _store("A")
    action = {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"],
        "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [{"claim_id": item["claim_id"], "assessment": "violates" if item["slot"] == 1 else "fits", "reason": item["selected_term"]} for item in manifest["slot_claims"]],
        "option_checks": [{"claim_id": item["claim_id"], "assessment": "fits" if item["option"] == "I" else "violates", "reason": item["option_text"]} for item in manifest["elimination_claims"]],
        "selected_option_check": {"assessment": "violates", "reason": manifest["selected_option_text"]},
        "summary": "selected option fails",
    }
    result = compiler.critic_verification_action(action)
    assert result["status"] == "need_fix"
    packet = __import__("json").loads(result["reason"])["repair_packet"]
    assert packet["forbidden_answers"] == ["A"]
    assert packet["viable_challengers"] == ["I"]


def test_assessment_schema_rejects_model_generated_claim_verdict():
    errors = validate_action(
        "critic",
        {"op": "audit_choice", "candidate_version": "result@v1", "manifest_digest": "x", "selected_option_check": {"verdict": "supported", "reason": "text"}, "slot_checks": [], "option_checks": [], "summary": "x"},
        task_type="multiple_choice",
        protocol_mode="semantic_delta_v2",
    )
    assert any("assessment" in error for error in errors)
    old_enum_errors = validate_action(
        "critic",
        {
            "op": "audit_choice", "candidate_version": "result@v1", "manifest_digest": "x",
            "selected_option_check": {"assessment": "compatible", "reason": "text"},
            "slot_checks": [{"claim_id": "result@v1:slot:1", "assessment": "compatible", "reason": "text"}],
            "option_checks": [{"claim_id": "result@v1:elimination:B", "assessment": "compatible", "reason": "text"}],
            "summary": "x",
        },
        task_type="multiple_choice",
        protocol_mode="semantic_delta_v2",
    )
    assert any("assessment" in error for error in old_enum_errors)


def test_insufficient_audit_preserves_structured_repair_packet():
    _store_obj, compiler, manifest = _store("A")
    action = {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"],
        "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [{"claim_id": item["claim_id"], "assessment": "unknown", "reason": item["selected_term"]} for item in manifest["slot_claims"]],
        "option_checks": [{"claim_id": item["claim_id"], "assessment": "unknown", "reason": item["option_text"]} for item in manifest["elimination_claims"]],
        "selected_option_check": {"assessment": "unknown", "reason": manifest["selected_option_text"]},
        "summary": "source audit is incomplete",
    }
    result = compiler.critic_verification_action(action)
    packet = __import__("json").loads(result["reason"])["repair_packet"]
    assert result["status"] == "uncertain"
    assert packet["required_change"] == "audited_claims"


def test_selected_fit_cannot_conflict_with_violated_slot():
    _store_obj, compiler, manifest = _store("A")
    action = {
        "op": "audit_choice",
        "candidate_version": manifest["candidate_version"],
        "manifest_digest": manifest["manifest_digest"],
        "slot_checks": [
            {"claim_id": item["claim_id"], "assessment": "violates" if item["slot"] == 1 else "fits", "reason": item["selected_term"]}
            for item in manifest["slot_claims"]
        ],
        "option_checks": [
            {"claim_id": item["claim_id"], "assessment": "fits" if item["option"] == "I" else "violates", "reason": item["option_text"]}
            for item in manifest["elimination_claims"]
        ],
        "selected_option_check": {"assessment": "fits", "reason": manifest["selected_option_text"]},
        "summary": "inconsistent aggregate",
    }
    result = compiler.critic_verification_action(action)
    assert result["status"] == "uncertain"
    assert result["error_type"] == "invalid_choice_audit_consistency"
