from scripts.run_mbpp_repair_context_ablation import (
    ARMS,
    build_diagnosis,
    repair_prompt,
    run_tests,
)


def _row():
    return {
        "question": "Write a Python function that satisfies the requirements.\n\nReturn x + 1.",
        "tests": [{"text": "assert add_one(1) == 2"}],
    }


def _diagnosis():
    failed = run_tests("def add_one(x):\n    return x", ["assert add_one(1) == 2"])
    assert not failed.passed
    return build_diagnosis(failed), failed.detail


def test_repair_context_arms_control_visible_payloads() -> None:
    diagnosis, execution = _diagnosis()
    previous_code = "def add_one(x):\n    return x"
    prompts = {
        arm: repair_prompt(
            _row(),
            arm=arm,
            diagnosis=diagnosis,
            previous_code=previous_code,
            execution_detail=execution,
            solver0_output=previous_code,
        )
        for arm in ARMS
    }

    assert '"status": "need_fix"' in prompts["binary"]
    assert "error_type" not in prompts["binary"]
    assert "error_type" in prompts["typed_error"]
    assert "concrete_failure" not in prompts["typed_error"]
    assert "concrete_failure" not in prompts["concrete_failure"]
    assert "FAILED TEST:" in prompts["concrete_failure"]
    assert "expected:" in prompts["concrete_failure"]
    assert "actual:" in prompts["concrete_failure"]
    assert "repair_instruction" not in prompts["diagnostic"]
    assert "must return 2" in prompts["diagnostic"]
    assert "Previous code:" not in prompts["diagnostic"]
    assert "Previous code:" in prompts["full_context"]
    assert "Execution detail:" in prompts["full_context"]
    assert "Original Solver0 raw output:" in prompts["full_context"]


def test_build_diagnosis_extracts_actionable_assertion_failure() -> None:
    diagnosis, _execution = _diagnosis()

    assert diagnosis["status"] == "need_fix"
    assert diagnosis["error_type"] == "wrong_value"
    assert "add_one" in diagnosis["error_location"]
    assert diagnosis["concrete_failure"]["expected"] == "2"
    assert diagnosis["concrete_failure"]["actual"] == "1"
    assert diagnosis["concrete_failure"]["input_repr"] == "1"
    assert diagnosis["repair_instruction"]
    assert "execution#execution_detail" in diagnosis["requested_fragments"]


def test_solver_generated_asserts_are_stripped_before_external_tests() -> None:
    failed = run_tests(
        "def add_one(x):\n    return x\n\nassert add_one(1) == 999",
        ["assert add_one(1) == 2"],
    )

    assert not failed.passed
    assert failed.detail["stage"] == "test"
    assert failed.detail["kind"] == "assertion_failure"
    assert failed.detail["failed_test"] == "assert add_one(1) == 2"
    assert failed.detail["expected"] == "2"
    assert failed.detail["actual"] == "1"
