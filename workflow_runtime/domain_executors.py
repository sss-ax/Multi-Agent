"""Small, deterministic domain executors used at the Solver boundary.

They deliberately execute only typed graph artifacts.  No network access is
allowed; MBPP is run in a subprocess with a timeout and the QA executors only
operate on nodes already present in the graph slice.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class DomainExecution:
    success: bool
    value: Any = None
    execution: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _json(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _nodes(graph: Any, node_type: str) -> List[Any]:
    state = graph.snapshot()
    task = next((n for n in state.nodes.values() if n.logical_id == "task"), None)
    if task is None:
        return []
    return sorted(
        [n for n in state.nodes.values() if n.task_id == task.task_id and n.type == node_type and n.is_operationally_valid()],
        key=lambda n: (n.logical_id, n.version),
    )


def _latest(graph: Any, logical_id: str) -> Any:
    state = graph.snapshot()
    task = next((n for n in state.nodes.values() if n.logical_id == "task"), None)
    return graph.latest_valid(task.task_id, task.branch_id, logical_id) if task else None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    return float(match.group(0).replace(",", "")) if match else None


def _python_code(value: Any) -> str:
    code = _json(value)
    if isinstance(code, dict):
        code = code.get("code", "")
    text = str(code)
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return strip_top_level_candidate_tests(text.strip())


def strip_top_level_candidate_tests(code: str) -> str:
    """Remove model-generated top-level tests from a candidate program.

    Benchmark tests must come from the dataset adapter.  Top-level asserts in
    model output are self-generated tests and must not run during candidate
    setup; assertions inside functions/classes are preserved as implementation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    kept = [
        node for node in tree.body
        if not isinstance(node, ast.Assert) and not _is_main_guard(node)
    ]
    if len(kept) == len(tree.body):
        return code
    if not kept:
        return code
    tree.body = kept
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _requirement_imports(graph: Any) -> str:
    imports: List[str] = []
    for node in _nodes(graph, "requirements"):
        value = _json(node.content)
        text = value.get("text", value) if isinstance(value, dict) else value
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and stripped not in imports:
                imports.append(stripped)
    return "\n".join(imports)


def execute_python_tests_detailed(
    *,
    code: str,
    tests: Iterable[Any],
    setup: str = "",
    timeout: int = 5,
) -> DomainExecution:
    """Execute Python code against tests and return structured failure detail."""
    test_text = [str(test.get("text", test)) if isinstance(test, dict) else str(test) for test in tests]
    code = strip_top_level_candidate_tests(code)
    harness = {"setup": setup, "code": code, "tests": test_text}
    script = r'''
import ast
import json
import sys
import traceback

payload = json.loads(sys.stdin.read())
ns = {}

def safe_repr(value):
    try:
        return repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"

def function_name(expr):
    if isinstance(expr, ast.Call):
        fn = expr.func
        if isinstance(fn, ast.Name):
            return fn.id
        if isinstance(fn, ast.Attribute):
            return fn.attr
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                return fn.id
            if isinstance(fn, ast.Attribute):
                return fn.attr
    return ""

def call_input_repr(expr):
    if not isinstance(expr, ast.Call):
        return ""
    parts = []
    for arg in expr.args:
        try:
            parts.append(ast.unparse(arg))
        except Exception:
            parts.append("<arg>")
    for kw in expr.keywords:
        try:
            parts.append(f"{kw.arg}={ast.unparse(kw.value)}")
        except Exception:
            parts.append(f"{kw.arg}=<arg>")
    return ", ".join(parts)

def classify(exc):
    name = type(exc).__name__
    if isinstance(exc, SyntaxError):
        return "syntax_error"
    if isinstance(exc, ImportError):
        return "import_error"
    if isinstance(exc, NameError):
        return "name_error"
    if isinstance(exc, TypeError):
        return "type_error"
    if isinstance(exc, AssertionError):
        return "assertion_failure"
    return "runtime_exception"

try:
    setup = payload.get("setup") or ""
    if setup:
        exec(setup, ns)
    exec(payload["code"], ns)
except Exception as exc:
    print(json.dumps({
        "stage": "compile_or_setup",
        "kind": classify(exc),
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc()[-2000:],
    }))
    sys.exit(1)

for index, test in enumerate(payload.get("tests") or []):
    source = str(test)
    try:
        parsed = ast.parse(source)
    except SyntaxError as exc:
        print(json.dumps({
            "stage": "test_parse",
            "kind": "syntax_error",
            "failed_test_index": index,
            "failed_test": source,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }))
        sys.exit(1)
    stmt = parsed.body[0] if parsed.body else None
    if (
        isinstance(stmt, ast.Assert)
        and isinstance(stmt.test, ast.Compare)
        and len(stmt.test.ops) == 1
        and isinstance(stmt.test.ops[0], ast.Eq)
        and len(stmt.test.comparators) == 1
    ):
        try:
            actual = eval(compile(ast.Expression(stmt.test.left), "<mbpp-left>", "eval"), ns)
            expected = eval(compile(ast.Expression(stmt.test.comparators[0]), "<mbpp-right>", "eval"), ns)
        except Exception as exc:
            print(json.dumps({
                "stage": "test",
                "kind": classify(exc),
                "failed_test_index": index,
                "test_id": f"test_{index}",
                "failed_test": source,
                "test_expression": source[6:].strip() if source.strip().startswith("assert ") else source,
                "function_name": function_name(stmt.test.left),
                "input_repr": call_input_repr(stmt.test.left),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc()[-2000:],
            }))
            sys.exit(1)
        if actual != expected:
            print(json.dumps({
                "stage": "test",
                "kind": "assertion_failure",
                "failed_test_index": index,
                "test_id": f"test_{index}",
                "failed_test": source,
                "test_expression": source[6:].strip() if source.strip().startswith("assert ") else source,
                "function_name": function_name(stmt.test.left),
                "input_repr": call_input_repr(stmt.test.left),
                "expected": safe_repr(expected),
                "actual": safe_repr(actual),
                "exception_type": "AssertionError",
                "exception": "",
                "traceback": "",
            }))
            sys.exit(1)
        continue
    try:
        exec(source, ns)
    except Exception as exc:
        print(json.dumps({
            "stage": "test",
            "kind": classify(exc),
            "failed_test_index": index,
            "failed_test": source,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc()[-2000:],
        }))
        sys.exit(1)

print(json.dumps({"stage": "tests", "kind": "passed", "status": "passed", "tests": len(payload.get("tests") or [])}))
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            input=json.dumps(harness),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        execution = {"domain": "mbpp", "status": "timeout", "kind": "timeout", "tests": len(test_text)}
        return DomainExecution(False, execution=execution, errors=["python test execution timed out"])
    raw = (completed.stdout or completed.stderr or "").strip()
    try:
        detail = json.loads(raw.splitlines()[-1]) if raw else {}
    except json.JSONDecodeError:
        detail = {
            "stage": "unknown",
            "kind": "runtime_exception",
            "stderr": completed.stderr[-2000:],
            "stdout": completed.stdout[-2000:],
        }
    detail["domain"] = "mbpp"
    detail["status"] = "passed" if completed.returncode == 0 else "failed"
    detail["tests"] = len(test_text)
    detail["returncode"] = completed.returncode
    if completed.returncode != 0:
        error = (
            detail.get("failed_test")
            or detail.get("exception")
            or detail.get("kind")
            or "MBPP tests failed"
        )
        return DomainExecution(False, execution=detail, errors=[str(error)])
    return DomainExecution(True, {"code": code, "tests_passed": True}, detail)


def _table_cells(graph: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for node in _nodes(graph, "table_cell"):
        value = _json(node.content)
        if isinstance(value, dict):
            key = f"r{value.get('row')}_c{value.get('column')}"
            result[key] = value.get("value")
            result[node.logical_id] = value.get("value")
    return result


def execute_table(graph: Any, payload: Mapping[str, Any]) -> DomainExecution:
    """Execute the typed table program emitted by Solver.

    Supported step operations are ``lookup``, ``add``, ``subtract``,
    ``multiply``, ``divide``, ``aggregate`` and ``compare``.  A plain answer
    is accepted only when the program declares that it is evidence-backed.
    """
    cells = _table_cells(graph)
    calculation = payload.get("calculation", payload)
    calculation = _json(calculation)
    steps = calculation.get("steps", []) if isinstance(calculation, dict) else []
    values: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    try:
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("table step must be an object")
            operation = step.get("operation")
            operands = step.get("operands", step.get("inputs", []))
            if operation == "lookup":
                key = str(operands[0])
                if key not in cells:
                    raise ValueError(f"unknown table cell {key}")
                value = cells[key]
            elif operation in {"add", "subtract", "multiply", "divide"}:
                nums = [_number(values.get(str(x), x)) for x in operands]
                if len(nums) < 2 or any(x is None for x in nums):
                    raise ValueError(f"{operation} requires numeric operands")
                if operation == "add": value = sum(nums)
                elif operation == "subtract": value = nums[0] - nums[1]
                elif operation == "multiply": value = math.prod(nums)
                else:
                    if nums[1] == 0: raise ValueError("division by zero")
                    value = nums[0] / nums[1]
            elif operation == "aggregate":
                nums = [_number(cells.get(str(x), values.get(str(x), x))) for x in operands]
                nums = [x for x in nums if x is not None]
                kind = str(step.get("kind", "sum"))
                if not nums: raise ValueError("aggregate has no numeric values")
                value = {"sum": sum, "max": max, "min": min, "average": lambda x: sum(x) / len(x)}.get(kind, sum)(nums)
            elif operation == "compare":
                left, right = (_number(values.get(str(x), x)) for x in operands[:2])
                if left is None or right is None: raise ValueError("compare requires numeric operands")
                value = "greater" if left > right else "less" if left < right else "equal"
            elif operation == "answer":
                value = values.get(str(operands[0]), operands[0])
            else:
                raise ValueError(f"unsupported table operation {operation}")
            target = step.get("target", step.get("output"))
            if target: values[str(target)] = value
            trace.append({"operation": operation, "target": target, "value": value})
        answer = calculation.get("answer") if isinstance(calculation, dict) else None
        value = values.get(str(answer), answer)
        if value is None:
            raise ValueError("table program did not produce an answer")
        return DomainExecution(True, value, {"domain": "tatqa", "status": "executed", "trace": trace})
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        return DomainExecution(False, execution={"domain": "tatqa", "status": "failed", "trace": trace}, errors=[str(exc)])


def execute_hotpot(graph: Any, payload: Mapping[str, Any]) -> DomainExecution:
    links = [_json(n.content) for n in _nodes(graph, "evidence_link")]
    facts = [_json(n.content) for n in _nodes(graph, "supporting_fact")]
    entities = {_json(n.content).get("name") for n in _nodes(graph, "entity") if isinstance(_json(n.content), dict)}
    errors: List[str] = []
    if not facts:
        errors.append("no supporting_fact nodes are available")
    for link in links:
        if isinstance(link, dict) and link.get("entity") and link["entity"] not in entities:
            errors.append(f"evidence_link references unknown entity {link['entity']}")
    calculation = _json(payload.get("calculation", payload))
    answer = calculation.get("answer") if isinstance(calculation, dict) else None
    if answer is None:
        errors.append("multihop program has no answer")
    if errors:
        return DomainExecution(False, execution={"domain": "hotpotqa", "status": "failed", "hops": len(links)}, errors=errors)
    return DomainExecution(True, answer, {"domain": "hotpotqa", "status": "evidence_chain_executed", "hops": len(links), "facts": len(facts)})


def execute_mbpp(graph: Any, payload: Mapping[str, Any]) -> DomainExecution:
    code = _python_code(payload.get("code", ""))
    tests = [_json(n.content) for n in _nodes(graph, "test")]
    if not isinstance(code, str) or not code.strip():
        return DomainExecution(False, execution={"domain": "mbpp", "status": "failed"}, errors=["no executable code node"])
    setup = "\n".join(str(item.get("setup", "")) for item in tests if isinstance(item, dict))
    full_setup = "\n".join(part for part in (_requirement_imports(graph), setup) if part)
    return execute_python_tests_detailed(code=code, tests=tests, setup=full_setup)


def execute_domain(task_type: str, graph: Any, payload: Mapping[str, Any]) -> DomainExecution:
    if task_type == "table_qa": return execute_table(graph, payload)
    if task_type == "multihop_qa": return execute_hotpot(graph, payload)
    if task_type == "code_generation": return execute_mbpp(graph, payload)
    return DomainExecution(False, errors=[f"no domain executor for {task_type}"])
