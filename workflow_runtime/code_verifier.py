"""Shared code artifact verification boundary for MBPP and HumanEval."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .domain_executors import DomainExecution, execute_python_tests_detailed
from .evaluation_context import EvaluationContext


@dataclass(frozen=True)
class CodeVerificationResult:
    status: str
    passed: bool
    public_test_count: int = 0
    private_test_count: int = 0
    failure_type: str = ""
    failure_summary: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "public_test_count": self.public_test_count,
            "private_test_count": self.private_test_count,
            "failure_type": self.failure_type,
            "failure_summary": self.failure_summary,
            "details": copy.deepcopy(dict(self.details)),
        }


class CodeVerifier:
    """Run public/private tests while returning only safe diagnostics."""

    def verify(
        self,
        code: str,
        *,
        public_tests: Iterable[Any] = (),
        private_tests: Iterable[Any] = (),
        setup: str = "",
        timeout: int = 5,
        evaluation: EvaluationContext | None = None,
    ) -> CodeVerificationResult:
        public = list(public_tests)
        private = list(private_tests)
        all_tests = [*public, *private]
        execution: DomainExecution = execute_python_tests_detailed(
            code=code,
            tests=all_tests,
            setup=setup,
            timeout=timeout,
        )
        has_private_tests = bool(private)
        detail = (
            evaluation.safe_verification_summary(
                execution.execution,
                private=has_private_tests,
            )
            if evaluation
            else self._safe_detail(execution.execution, private=has_private_tests)
        )
        if execution.success:
            return CodeVerificationResult(
                status="passed",
                passed=True,
                public_test_count=len(public),
                private_test_count=len(private),
                details=detail,
            )
        failure_type = str(execution.execution.get("kind", "verification_failure"))
        # Never echo a failing private assertion.  The model gets a stable,
        # typed signal and can repair its artifact without learning hidden
        # evaluator material.
        summary = (
            f"{failure_type} in private verification"
            if has_private_tests
            else str(execution.errors[0] if execution.errors else failure_type)[:240]
        )
        return CodeVerificationResult(
            status="failed",
            passed=False,
            public_test_count=len(public),
            private_test_count=len(private),
            failure_type=failure_type,
            failure_summary=summary,
            details=detail,
        )

    @staticmethod
    def _safe_detail(detail: Mapping[str, Any], *, private: bool = False) -> dict[str, Any]:
        forbidden = {
            "failed_test", "test_expression", "test_source", "tests",
            "hidden_tests", "official_tests", "gold_answer", "traceback",
            "stdout", "stderr",
        }
        if private:
            forbidden.update({"expected", "actual", "input", "input_repr", "error", "errors"})
        result: dict[str, Any] = {}
        for key, value in detail.items():
            if str(key).casefold() in forbidden:
                continue
            if isinstance(value, str):
                result[key] = value[:240]
            elif isinstance(value, (int, float, bool)) or value is None:
                result[key] = value
        return result
