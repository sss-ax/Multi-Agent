from workflow_runtime.evaluator import TaskEvaluator, parse_reference_answer


def test_numeric_reference_uses_tolerance():
    result = TaskEvaluator("numeric_solve", "5", tolerance=1e-3).evaluate(5.0005)
    assert result.status == "evaluated"
    assert result.correct is True


def test_numeric_reference_accepts_explicit_answer_in_natural_language():
    result = TaskEvaluator("numeric_solve", 5).evaluate(
        "The calculation is complete. FINAL ANSWER: The total is 5."
    )
    assert result.numeric_match is True
    assert result.correct is True
    assert result.numeric_match is True
    assert result.comparison == "numeric_tolerance"


def test_numeric_reference_accepts_terminal_punctuation():
    result = TaskEvaluator("numeric_solve", 5).evaluate("5.")
    assert result.numeric_match is True
    assert result.correct is True


def test_numeric_reference_rejects_wrong_answer():
    result = TaskEvaluator("numeric_solve", 5).evaluate(6)
    assert result.correct is False
    assert result.exact_match is False
    assert result.numeric_match is False


def test_text_reference_is_normalized():
    result = TaskEvaluator("multihop_qa", "Paris").evaluate("  paris\n")
    assert result.correct is True
    assert result.normalized_match is True


def test_structured_reference_ignores_object_key_order():
    result = TaskEvaluator("code_generation", '{"tests_passed": true, "code": "x"}').evaluate(
        {"code": "x", "tests_passed": True}
    )
    assert result.correct is True
    assert result.comparison == "canonical_json"


def test_missing_reference_is_not_counted_as_incorrect():
    result = TaskEvaluator("numeric_solve").evaluate(5)
    assert result.status == "not_evaluated"
    assert result.correct is None
    assert parse_reference_answer("true") is True
