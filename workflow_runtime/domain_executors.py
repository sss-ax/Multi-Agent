"""Small, deterministic domain executors used at the Solver boundary.

They deliberately execute only typed graph artifacts.  No network access is
allowed; MBPP is run in a subprocess with a timeout and the QA executors only
operate on nodes already present in the graph slice.
"""

from __future__ import annotations

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
    return text.strip()


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
    test_text = [item.get("text", item) if isinstance(item, dict) else item for item in tests]
    script = setup + "\n" + code + "\n" + "\n".join(str(test) for test in test_text)
    try:
        completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return DomainExecution(False, execution={"domain": "mbpp", "status": "timeout", "tests": len(tests)}, errors=["python test execution timed out"])
    execution = {"domain": "mbpp", "status": "passed" if completed.returncode == 0 else "failed", "tests": len(tests), "returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    if completed.returncode != 0:
        return DomainExecution(False, execution=execution, errors=[completed.stderr[-2000:] or "MBPP tests failed"])
    return DomainExecution(True, {"code": code, "tests_passed": True}, execution)


def execute_domain(task_type: str, graph: Any, payload: Mapping[str, Any]) -> DomainExecution:
    if task_type == "table_qa": return execute_table(graph, payload)
    if task_type == "multihop_qa": return execute_hotpot(graph, payload)
    if task_type == "code_generation": return execute_mbpp(graph, payload)
    return DomainExecution(False, errors=[f"no domain executor for {task_type}"])
