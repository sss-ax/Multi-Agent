"""Evaluate the graph-only workflow on common deterministic benchmarks.

Supported domains:
- gsm8k: numeric exact match after answer extraction.
- mbpp: generated Python is executed against bundled tests.
- humaneval: generated Python is executed against HumanEval check().
- mmlu_pro: multiple-choice exact match against the gold letter.
- hotpotqa: answer exact match and token F1 using the official normalization.
- tatqa: numeric/text table QA match.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _raw_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
        with _open_text(path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.values() if isinstance(data, dict) else data
        for row in rows:
            if isinstance(row, dict):
                yield row
        return
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"unsupported data file extension: {path}")


def _answer_letter(row: Dict[str, Any], options: List[str]) -> str:
    value = row.get("answer", row.get("gold_answer", row.get("label", row.get("target"))))
    if isinstance(value, int):
        return LETTERS[value]
    text = str(value).strip()
    if re.fullmatch(r"[A-Ja-j]", text):
        return text.upper()
    if text.isdigit():
        return LETTERS[int(text)]
    for index, option in enumerate(options):
        if normalize_answer(option) == normalize_answer(text):
            return LETTERS[index]
    return text.upper()


def _options(row: Dict[str, Any]) -> List[str]:
    options = row.get("options", row.get("choices", row.get("options_")))
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
            options = parsed
        except json.JSONDecodeError:
            options = [item.strip() for item in re.split(r"\s*\|\s*", options) if item.strip()]
    if isinstance(options, dict):
        return [str(options[key]) for key in sorted(options)]
    if isinstance(options, list):
        return [str(item) for item in options]
    result = []
    for letter in LETTERS[:10]:
        for key in (letter, letter.lower(), f"option_{letter}", f"option_{letter.lower()}"):
            if key in row and str(row[key]).strip():
                result.append(str(row[key]))
                break
    return result


def _hotpot_context(row: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    context = row.get("context", [])
    entities: List[str] = []
    facts: List[Dict[str, Any]] = []
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_by_title = context.get("sentences", [])
        context = list(zip(titles, sentences_by_title))
    for item in context:
        if isinstance(item, list) and len(item) >= 2:
            title, sentences = str(item[0]), item[1]
        elif isinstance(item, tuple) and len(item) >= 2:
            title, sentences = str(item[0]), item[1]
        elif isinstance(item, dict):
            title, sentences = str(item.get("title", "")), item.get("sentences", item.get("text", []))
        else:
            continue
        entities.append(title)
        if isinstance(sentences, str):
            sentences = [sentences]
        for sent_id, sentence in enumerate(sentences or []):
            facts.append({"title": title, "sent_id": sent_id, "text": str(sentence)})
    supporting = []
    raw_supporting = row.get("supporting_facts", [])
    if isinstance(raw_supporting, dict):
        raw_supporting = list(zip(raw_supporting.get("title", []), raw_supporting.get("sent_id", [])))
    for item in raw_supporting:
        if isinstance(item, list) and len(item) >= 2:
            supporting.append({"entity": str(item[0]), "sent_id": item[1]})
        elif isinstance(item, tuple) and len(item) >= 2:
            supporting.append({"entity": str(item[0]), "sent_id": item[1]})
        elif isinstance(item, dict):
            supporting.append(item)
    return entities, facts, supporting


def render_native_task(row: Dict[str, Any]) -> str:
    task_type = row["task_type"]
    if task_type == "code_generation":
        requirements = row.get("requirements", {})
        text = requirements.get("text", row.get("question", "")) if isinstance(requirements, dict) else str(requirements)
        entry_point = requirements.get("entry_point", "") if isinstance(requirements, dict) else ""
        tests = row.get("tests", [])
        test_lines = [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in tests]
        lines = [
            "Write Python code that passes the tests.",
            "Return only executable Python code, with no Markdown.",
        ]
        if entry_point:
            lines.append(f"The required function name is: {entry_point}")
        lines.extend(["", "Requirements:", str(text).strip()])
        if test_lines:
            lines.extend(["", "Tests that your code must pass:", *test_lines])
        return "\n".join(lines)
    if task_type == "multihop_qa":
        lines = [str(row["question"]).strip(), "", "Evidence:"]
        for fact in row.get("supporting_facts", row.get("evidence", [])):
            if isinstance(fact, dict):
                title = fact.get("title", fact.get("entity", ""))
                text = fact.get("text", fact)
                lines.append(f"- {title}: {text}")
            else:
                lines.append(f"- {fact}")
        return "\n".join(lines)
    return str(row["question"]).strip()


def read(path: Path, domain: str, limit: int) -> List[Dict[str, Any]]:
    result = []
    for raw in _raw_rows(path):
        row = dict(raw)
        if domain == "gsm8k":
            answer = str(row.get("answer", ""))
            match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer)
            if not match:
                continue
            row = {
                "sample_id": row.get("id", f"gsm8k_eval_{len(result)}"),
                "domain": "gsm8k",
                "task_type": "numeric_solve",
                "question": str(row.get("question", "")).strip(),
                "gold_answer": match.group(1).replace(",", ""),
            }
        elif domain == "humaneval":
            prompt = str(row.get("prompt", ""))
            entry_point = str(row.get("entry_point", ""))
            test = str(row.get("test", ""))
            row = {
                "sample_id": row.get("task_id", f"humaneval_{len(result)}"),
                "domain": "humaneval",
                "task_type": "code_generation",
                "question": (
                    "Complete the following Python function. Return only executable Python code.\n\n"
                    f"{prompt}"
                ),
                "requirements": {"text": prompt, "entry_point": entry_point},
                "tests": [{"setup": "", "text": f"{test}\ncheck({entry_point})"}],
                "gold_answer": {"entry_point": entry_point},
            }
        elif domain == "mbpp":
            tests = row.get("tests", row.get("test_list", []))
            if isinstance(tests, str):
                try:
                    tests = json.loads(tests)
                except json.JSONDecodeError:
                    tests = [line for line in tests.splitlines() if line.strip()]
            row = {
                "sample_id": row.get("task_id", row.get("id", f"mbpp_{len(result)}")),
                "domain": "mbpp",
                "task_type": "code_generation",
                "question": (
                    "Write a Python function that satisfies the requirements. "
                    "Return only executable Python code.\n\n"
                    f"{row.get('text', row.get('question', row.get('prompt', '')))}"
                ),
                "requirements": {"text": row.get("text", row.get("question", row.get("prompt", "")))},
                "tests": [{"setup": "", "text": test} for test in tests],
                "gold_answer": {"tests": len(tests)},
            }
        elif domain == "mmlu_pro":
            options = _options(row)
            if not options:
                continue
            gold = _answer_letter(row, options)
            choice_lines = [f"{LETTERS[index]}. {option}" for index, option in enumerate(options)]
            row = {
                "sample_id": row.get("question_id", row.get("id", f"mmlu_pro_{len(result)}")),
                "domain": "mmlu_pro",
                "task_type": "multiple_choice",
                "question": (
                    f"{row.get('question', row.get('input', ''))}\n\n"
                    + "\n".join(choice_lines)
                    + "\n\nAnswer with only the option letter."
                ),
                "choices": [{"label": LETTERS[index], "text": option} for index, option in enumerate(options)],
                "gold_answer": gold,
                "category": row.get("category", row.get("subject", "")),
            }
        elif domain == "hotpotqa":
            entities, facts, links = _hotpot_context(row)
            row = {
                "sample_id": row.get("_id", row.get("id", f"hotpotqa_{len(result)}")),
                "domain": "hotpotqa",
                "task_type": "multihop_qa",
                "question": str(row.get("question", "")).strip(),
                "entities": entities,
                "supporting_facts": facts,
                "evidence_links": links,
                "gold_answer": row.get("answer", ""),
            }
        result.append(row)
        if limit and len(result) >= limit:
            break
    return result


def add_source_nodes(workflow: Any, graph: Any, row: Dict[str, Any], sample_id: str) -> None:
    task_type = row["task_type"]
    refs: Dict[str, Any] = {}

    def add(logical_id: str, node_type: str, content: Any) -> None:
        refs[logical_id] = graph.add_node(
            task_id=sample_id,
            branch_id="main",
            logical_id=logical_id,
            node_type=node_type,
            content=content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, separators=(",", ":")),
            owner="dataset",
            status="ready",
            validation={"schema_valid": True},
            created_by_role="dataset",
        )

    if task_type == "table_qa":
        add("table", "table", row.get("table", {}))
        for index, cell in enumerate(row.get("table_cells", row.get("facts", [])), start=1):
            add(f"table_cell_{index}", "table_cell", cell)
        for index, evidence in enumerate(row.get("evidence", []), start=1):
            add(f"evidence_{index}", "evidence", evidence)
    elif task_type == "multihop_qa":
        for index, entity in enumerate(row.get("entities", []), start=1):
            add(f"entity_{index}", "entity", {"name": entity})
        for index, fact in enumerate(row.get("supporting_facts", row.get("evidence", [])), start=1):
            add(f"supporting_fact_{index}", "supporting_fact", fact)
        for index, link in enumerate(row.get("evidence_links", []), start=1):
            add(f"evidence_link_{index}", "evidence_link", link)
    elif task_type == "multiple_choice":
        choices = row.get("choices", [])
        add("choice_schema", "choice_schema", {
            "allowed_labels": [str(choice.get("label", "")).strip().upper() for choice in choices if isinstance(choice, dict)],
        })
        for index, choice in enumerate(choices, start=1):
            add(f"choice_{index}", "choice", choice)
    elif task_type == "code_generation":
        add("requirements", "requirements", row.get("requirements", {"text": row["question"]}))
        for index, test in enumerate(row.get("tests", []), start=1):
            add(f"test_{index}", "test", test)

    task = graph.latest_valid(sample_id, "main", "task")
    for node in refs.values():
        graph.add_edge(source=task.node_id, target=node.node_id, relation="depends_on", created_by_role="dataset")


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def hotpot_normalize(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def hotpot_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = hotpot_normalize(prediction).split()
    gold_tokens = hotpot_normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_structured_answer(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return extract_structured_answer(parsed)
    if isinstance(value, dict):
        if "value" in value:
            return extract_structured_answer(value["value"])
        for key in ("answer", "final_answer", "choice", "solver_choice"):
            if key in value:
                return extract_structured_answer(value[key])
    return value


def extract_choice(value: Any) -> str:
    value = extract_structured_answer(value)
    text = str(value).strip()
    match = re.search(r"\b([A-J])\b", text.upper())
    return match.group(1) if match else text[:1].upper()


def extract_code(value: Any) -> str:
    text = str(value or "")
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.lstrip().startswith(("def ", "class ", "from ", "import "))), 0)
    code = "\n".join(lines[start:]).strip()
    return code


def prompt_imports(requirements: Any) -> str:
    text = ""
    if isinstance(requirements, dict):
        text = str(requirements.get("text", ""))
    else:
        text = str(requirements or "")
    imports = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and stripped not in imports:
            imports.append(stripped)
    return "\n".join(imports)


def run_code_tests(code: str, row: Dict[str, Any]) -> Dict[str, Any]:
    tests = row.get("tests", [])
    setup = "\n".join(str(item.get("setup", "")) for item in tests if isinstance(item, dict))
    test_text = [item.get("text", item) if isinstance(item, dict) else item for item in tests]
    imports = prompt_imports(row.get("requirements"))
    script = "\n".join(part for part in (imports, setup, code, "\n".join(str(test) for test in test_text)) if part)
    try:
        completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return {"correct": False, "pass_at_1": 0.0, "predicted_answer": {"code": code, "tests_passed": False, "errors": ["python test execution timed out"]}}
    passed = completed.returncode == 0
    error = completed.stderr[-2000:] or completed.stdout[-2000:]
    return {
        "correct": passed,
        "pass_at_1": float(passed),
        "predicted_answer": {"code": code, "tests_passed": passed, "errors": [] if passed else [error]},
    }


def numeric_answer(value: Any) -> str:
    text = normalize_answer(value).replace(",", "")
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else f"{number:.8f}".rstrip("0").rstrip(".")
    except ValueError:
        return text


def extract_numeric_answer(value: Any) -> str:
    text = normalize_answer(value).replace(",", "")
    labelled = re.findall(
        r"(?:final\s+answer|answer|result|total|答案|结果|总数|makes|is)"
        r"[^0-9+\-]{0,80}"
        r"([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    candidates = labelled or re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not candidates:
        return text
    return numeric_answer(candidates[-1])


def is_correct(workflow: Any, row: Dict[str, Any], graph: Any) -> bool:
    task_type = row["task_type"]
    result = workflow.latest_node_by_type(graph, "result")
    if task_type == "code_generation":
        if result is None:
            return False
        try:
            value = json.loads(result.content)
            return bool(value.get("value", {}).get("tests_passed"))
        except (TypeError, json.JSONDecodeError, AttributeError):
            return False
    final = workflow.latest_node_by_type(graph, "final_answer")
    if final is None:
        return False
    predicted_value = extract_structured_answer(final.content)
    predicted = normalize_answer(predicted_value)
    gold = normalize_answer(row.get("gold_answer", ""))
    if task_type == "numeric_solve":
        return extract_numeric_answer(predicted) == numeric_answer(gold)
    if task_type == "multiple_choice":
        return extract_choice(predicted) == extract_choice(gold)
    if task_type == "table_qa":
        try:
            return abs(float(predicted.replace(",", "")) - float(gold.replace(",", ""))) < 1e-6
        except ValueError:
            return predicted == gold
    return predicted == gold


def score_record(workflow: Any, row: Dict[str, Any], graph: Any) -> Dict[str, Any]:
    task_type = row["task_type"]
    result = workflow.latest_node_by_type(graph, "result")
    final = workflow.latest_node_by_type(graph, "final_answer")
    predicted = extract_structured_answer(final.content) if final is not None else None
    if task_type == "code_generation":
        passed = False
        if result is not None:
            try:
                value = json.loads(result.content) if isinstance(result.content, str) else result.content
                passed = bool(value.get("value", value).get("tests_passed"))
            except (TypeError, json.JSONDecodeError, AttributeError):
                passed = False
        return {"correct": passed, "pass_at_1": float(passed), "predicted_answer": predicted}
    if task_type == "multihop_qa":
        em = float(hotpot_normalize(predicted) == hotpot_normalize(row.get("gold_answer", "")))
        f1 = hotpot_f1(predicted, row.get("gold_answer", ""))
        return {"correct": bool(em), "exact_match": em, "f1": f1, "predicted_answer": predicted}
    correct = is_correct(workflow, row, graph)
    return {"correct": correct, "exact_match": float(correct), "predicted_answer": predicted}


def score_prediction(row: Dict[str, Any], predicted: Any) -> Dict[str, Any]:
    task_type = row["task_type"]
    predicted = extract_structured_answer(predicted)
    if task_type == "code_generation":
        return run_code_tests(extract_code(predicted), row)
    if task_type == "multihop_qa":
        em = float(hotpot_normalize(predicted) == hotpot_normalize(row.get("gold_answer", "")))
        f1 = hotpot_f1(predicted, row.get("gold_answer", ""))
        return {"correct": bool(em), "exact_match": em, "f1": f1, "predicted_answer": predicted}
    if task_type == "multiple_choice":
        correct = extract_choice(predicted) == extract_choice(row.get("gold_answer", ""))
        return {"correct": correct, "exact_match": float(correct), "predicted_answer": predicted}
    gold = normalize_answer(row.get("gold_answer", ""))
    pred = normalize_answer(predicted)
    if task_type == "numeric_solve":
        correct = extract_numeric_answer(predicted) == numeric_answer(gold)
    elif task_type == "table_qa":
        try:
            correct = abs(float(pred.replace(",", "")) - float(gold.replace(",", ""))) < 1e-6
        except ValueError:
            correct = pred == gold
    else:
        correct = pred == gold
    return {"correct": correct, "exact_match": float(correct), "predicted_answer": predicted}


def communication_totals(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "graph_delta_candidate_tokens": 0,
        "graph_delta_sent_tokens": 0,
        "graph_delta_sent_nodes": 0,
        "graph_communication_events": 0,
        "graph_delta_mandatory_roots": 0,
        "graph_delta_optional_roots": 0,
        "graph_delta_selected_optional_roots": 0,
        "graph_delta_mandatory_root_tokens": 0,
        "graph_delta_optional_root_tokens": 0,
        "graph_delta_selected_optional_root_tokens": 0,
        "semantic_nack_count": 0,
        "semantic_hard_nack_count": 0,
        "semantic_soft_nack_count": 0,
        "semantic_verification_nack_count": 0,
        "semantic_quality_nack_count": 0,
        "semantic_initial_nack_count": 0,
        "semantic_initial_hard_nack_count": 0,
        "semantic_initial_soft_nack_count": 0,
        "semantic_initial_verification_nack_count": 0,
        "semantic_initial_quality_nack_count": 0,
        "semantic_missing_repaired_count": 0,
        "semantic_hard_repaired_count": 0,
        "semantic_soft_repaired_count": 0,
        "semantic_verification_repaired_count": 0,
        "semantic_quality_repaired_count": 0,
        "feedback_sent_tokens": 0,
        "feedback_transport_tokens": 0,
        "feedback_newly_visible_tokens": 0,
        "feedback_newly_rendered_tokens": 0,
        "total_comm_tokens": 0,
        "core_comm_tokens": 0,
        "delta_comm_tokens": 0,
        "verification_comm_tokens": 0,
        "quality_comm_tokens": 0,
        "control_comm_tokens": 0,
        "unique_comm_tokens": 0,
        "repeated_comm_tokens": 0,
        "receiver_seen_hit_count": 0,
        "state_delta_tokens": 0,
        "full_state_equivalent_tokens": 0,
        "revision_success_count": 0,
        "revision_regression_count": 0,
        "communication_token_accounting_errors": 0,
        "communication_token_breakdown_errors": 0,
        "feedback_render_accounting_errors": 0,
        "unique_repeated_accounting_errors": 0,
        "targeted_refinement_skipped_count": 0,
        "quality_refinement_budget_exceeded_count": 0,
    }
    by_pair: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for action_log in record.get("runtime_summary", {}).get("records", []):
            for event in action_log.get("communication", []):
                pair = f"{event.get('sender', '')}->{event.get('receiver', '')}"
                pair_totals = by_pair.setdefault(pair, {key: 0 for key in totals})
                totals["graph_communication_events"] += 1
                totals["graph_delta_candidate_tokens"] += int(event.get("candidate_tokens", 0) or 0)
                totals["graph_delta_sent_tokens"] += int(event.get("sent_tokens", 0) or 0)
                totals["graph_delta_sent_nodes"] += int(event.get("sent_count", 0) or 0)
                totals["graph_delta_mandatory_roots"] += int(event.get("mandatory_root_count", 0) or 0)
                totals["graph_delta_optional_roots"] += int(event.get("optional_root_count", 0) or 0)
                totals["graph_delta_selected_optional_roots"] += int(event.get("selected_optional_root_count", 0) or 0)
                totals["graph_delta_mandatory_root_tokens"] += int(event.get("mandatory_root_tokens", 0) or 0)
                totals["graph_delta_optional_root_tokens"] += int(event.get("optional_root_tokens", 0) or 0)
                totals["graph_delta_selected_optional_root_tokens"] += int(event.get("selected_optional_root_tokens", 0) or 0)
                totals["semantic_nack_count"] += int(bool(event.get("semantic_nack", False)))
                totals["semantic_hard_nack_count"] += int(bool(event.get("semantic_hard_nack", False)))
                totals["semantic_soft_nack_count"] += int(bool(event.get("semantic_soft_nack", False)))
                totals["semantic_verification_nack_count"] += int(bool(event.get("semantic_verification_nack", False)))
                totals["semantic_quality_nack_count"] += int(bool(event.get("semantic_quality_nack", False)))
                totals["semantic_initial_nack_count"] += int(bool(event.get("semantic_initial_nack", False)))
                totals["semantic_initial_hard_nack_count"] += int(bool(event.get("semantic_initial_hard_nack", False)))
                totals["semantic_initial_soft_nack_count"] += int(bool(event.get("semantic_initial_soft_nack", False)))
                totals["semantic_initial_verification_nack_count"] += int(bool(event.get("semantic_initial_verification_nack", False)))
                totals["semantic_initial_quality_nack_count"] += int(bool(event.get("semantic_initial_quality_nack", False)))
                for key in (
                    "semantic_missing_repaired_count",
                    "semantic_hard_repaired_count",
                    "semantic_soft_repaired_count",
                    "semantic_verification_repaired_count",
                    "semantic_quality_repaired_count",
                    "feedback_sent_tokens",
                    "feedback_transport_tokens",
                    "feedback_newly_visible_tokens",
                    "feedback_newly_rendered_tokens",
                    "total_comm_tokens",
                    "core_comm_tokens",
                    "delta_comm_tokens",
                    "verification_comm_tokens",
                    "quality_comm_tokens",
                    "control_comm_tokens",
                    "unique_comm_tokens",
                    "repeated_comm_tokens",
                    "receiver_seen_hit_count",
                    "state_delta_tokens",
                    "full_state_equivalent_tokens",
                    "revision_success_count",
                    "revision_regression_count",
                    "targeted_refinement_skipped_count",
                    "quality_refinement_budget_exceeded_count",
                ):
                    totals[key] += int(event.get(key, 0) or 0)
                if not bool(event.get("communication_token_accounting_ok", True)):
                    totals["communication_token_accounting_errors"] += 1
                if not bool(event.get("communication_token_breakdown_ok", True)):
                    totals["communication_token_breakdown_errors"] += 1
                if not bool(event.get("feedback_render_accounting_ok", True)):
                    totals["feedback_render_accounting_errors"] += 1
                if not bool(event.get("unique_repeated_accounting_ok", True)):
                    totals["unique_repeated_accounting_errors"] += 1
                pair_totals["graph_communication_events"] += 1
                pair_totals["graph_delta_candidate_tokens"] += int(event.get("candidate_tokens", 0) or 0)
                pair_totals["graph_delta_sent_tokens"] += int(event.get("sent_tokens", 0) or 0)
                pair_totals["graph_delta_sent_nodes"] += int(event.get("sent_count", 0) or 0)
                pair_totals["graph_delta_mandatory_roots"] += int(event.get("mandatory_root_count", 0) or 0)
                pair_totals["graph_delta_optional_roots"] += int(event.get("optional_root_count", 0) or 0)
                pair_totals["graph_delta_selected_optional_roots"] += int(event.get("selected_optional_root_count", 0) or 0)
                pair_totals["graph_delta_mandatory_root_tokens"] += int(event.get("mandatory_root_tokens", 0) or 0)
                pair_totals["graph_delta_optional_root_tokens"] += int(event.get("optional_root_tokens", 0) or 0)
                pair_totals["graph_delta_selected_optional_root_tokens"] += int(event.get("selected_optional_root_tokens", 0) or 0)
                pair_totals["semantic_nack_count"] += int(bool(event.get("semantic_nack", False)))
                pair_totals["semantic_hard_nack_count"] += int(bool(event.get("semantic_hard_nack", False)))
                pair_totals["semantic_soft_nack_count"] += int(bool(event.get("semantic_soft_nack", False)))
                pair_totals["semantic_verification_nack_count"] += int(bool(event.get("semantic_verification_nack", False)))
                pair_totals["semantic_quality_nack_count"] += int(bool(event.get("semantic_quality_nack", False)))
                pair_totals["semantic_initial_nack_count"] += int(bool(event.get("semantic_initial_nack", False)))
                pair_totals["semantic_initial_hard_nack_count"] += int(bool(event.get("semantic_initial_hard_nack", False)))
                pair_totals["semantic_initial_soft_nack_count"] += int(bool(event.get("semantic_initial_soft_nack", False)))
                pair_totals["semantic_initial_verification_nack_count"] += int(bool(event.get("semantic_initial_verification_nack", False)))
                pair_totals["semantic_initial_quality_nack_count"] += int(bool(event.get("semantic_initial_quality_nack", False)))
                for key in (
                    "semantic_missing_repaired_count",
                    "semantic_hard_repaired_count",
                    "semantic_soft_repaired_count",
                    "semantic_verification_repaired_count",
                    "semantic_quality_repaired_count",
                    "feedback_sent_tokens",
                    "feedback_transport_tokens",
                    "feedback_newly_visible_tokens",
                    "feedback_newly_rendered_tokens",
                    "total_comm_tokens",
                    "core_comm_tokens",
                    "delta_comm_tokens",
                    "verification_comm_tokens",
                    "quality_comm_tokens",
                    "control_comm_tokens",
                    "unique_comm_tokens",
                    "repeated_comm_tokens",
                    "receiver_seen_hit_count",
                    "state_delta_tokens",
                    "full_state_equivalent_tokens",
                    "revision_success_count",
                    "revision_regression_count",
                    "targeted_refinement_skipped_count",
                    "quality_refinement_budget_exceeded_count",
                ):
                    pair_totals[key] += int(event.get(key, 0) or 0)
                if not bool(event.get("communication_token_accounting_ok", True)):
                    pair_totals["communication_token_accounting_errors"] += 1
                if not bool(event.get("communication_token_breakdown_ok", True)):
                    pair_totals["communication_token_breakdown_errors"] += 1
                if not bool(event.get("feedback_render_accounting_ok", True)):
                    pair_totals["feedback_render_accounting_errors"] += 1
                if not bool(event.get("unique_repeated_accounting_ok", True)):
                    pair_totals["unique_repeated_accounting_errors"] += 1
    feedback_sent = int(totals.get("feedback_sent_tokens", 0) or 0)
    totals["feedback_utilization"] = (
        int(totals.get("feedback_newly_rendered_tokens", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    totals["feedback_nack_repair_efficiency"] = (
        int(totals.get("semantic_missing_repaired_count", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    totals["feedback_hard_repair_efficiency"] = (
        int(totals.get("semantic_hard_repaired_count", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    totals["feedback_soft_repair_efficiency"] = (
        int(totals.get("semantic_soft_repaired_count", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    totals["feedback_verification_repair_efficiency"] = (
        int(totals.get("semantic_verification_repaired_count", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    totals["feedback_quality_repair_efficiency"] = (
        int(totals.get("semantic_quality_repaired_count", 0) or 0) / feedback_sent
        if feedback_sent else 0.0
    )
    total_comm = int(totals.get("total_comm_tokens", 0) or 0)
    repeated_comm = int(totals.get("repeated_comm_tokens", 0) or 0)
    state_delta = int(totals.get("state_delta_tokens", 0) or 0)
    full_state = int(totals.get("full_state_equivalent_tokens", 0) or 0)
    totals["duplicate_ratio"] = repeated_comm / total_comm if total_comm else 0.0
    totals["incremental_saving"] = 1.0 - (state_delta / full_state) if full_state else 0.0
    totals["communication_token_breakdown_ok"] = total_comm == sum(
        int(totals.get(key, 0) or 0)
        for key in (
            "core_comm_tokens",
            "delta_comm_tokens",
            "verification_comm_tokens",
            "quality_comm_tokens",
            "control_comm_tokens",
        )
    )
    totals["feedback_render_accounting_ok"] = (
        int(totals.get("feedback_sent_tokens", 0) or 0)
        >= int(totals.get("feedback_newly_rendered_tokens", 0) or 0)
    )
    totals["unique_repeated_accounting_ok"] = total_comm == (
        int(totals.get("unique_comm_tokens", 0) or 0)
        + int(totals.get("repeated_comm_tokens", 0) or 0)
    )
    totals["communication_round_count"] = int(totals.get("graph_communication_events", 0) or 0)
    totals["communication_cost_tokens"] = int(totals.get("total_comm_tokens", 0) or 0)
    for pair_totals in by_pair.values():
        pair_feedback_sent = int(pair_totals.get("feedback_sent_tokens", 0) or 0)
        pair_totals["feedback_utilization"] = (
            int(pair_totals.get("feedback_newly_rendered_tokens", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_totals["feedback_nack_repair_efficiency"] = (
            int(pair_totals.get("semantic_missing_repaired_count", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_totals["feedback_hard_repair_efficiency"] = (
            int(pair_totals.get("semantic_hard_repaired_count", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_totals["feedback_soft_repair_efficiency"] = (
            int(pair_totals.get("semantic_soft_repaired_count", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_totals["feedback_verification_repair_efficiency"] = (
            int(pair_totals.get("semantic_verification_repaired_count", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_totals["feedback_quality_repair_efficiency"] = (
            int(pair_totals.get("semantic_quality_repaired_count", 0) or 0) / pair_feedback_sent
            if pair_feedback_sent else 0.0
        )
        pair_total_comm = int(pair_totals.get("total_comm_tokens", 0) or 0)
        pair_repeated_comm = int(pair_totals.get("repeated_comm_tokens", 0) or 0)
        pair_state_delta = int(pair_totals.get("state_delta_tokens", 0) or 0)
        pair_full_state = int(pair_totals.get("full_state_equivalent_tokens", 0) or 0)
        pair_totals["duplicate_ratio"] = (
            pair_repeated_comm / pair_total_comm if pair_total_comm else 0.0
        )
        pair_totals["incremental_saving"] = (
            1.0 - (pair_state_delta / pair_full_state) if pair_full_state else 0.0
        )
        pair_totals["communication_token_breakdown_ok"] = pair_total_comm == sum(
            int(pair_totals.get(key, 0) or 0)
            for key in (
                "core_comm_tokens",
                "delta_comm_tokens",
                "verification_comm_tokens",
                "quality_comm_tokens",
                "control_comm_tokens",
            )
        )
        pair_totals["feedback_render_accounting_ok"] = (
            int(pair_totals.get("feedback_sent_tokens", 0) or 0)
            >= int(pair_totals.get("feedback_newly_rendered_tokens", 0) or 0)
        )
        pair_totals["unique_repeated_accounting_ok"] = pair_total_comm == (
            int(pair_totals.get("unique_comm_tokens", 0) or 0)
            + int(pair_totals.get("repeated_comm_tokens", 0) or 0)
        )
        pair_totals["communication_round_count"] = int(pair_totals.get("graph_communication_events", 0) or 0)
        pair_totals["communication_cost_tokens"] = int(pair_totals.get("total_comm_tokens", 0) or 0)
    totals["graph_delta_by_pair"] = by_pair
    return totals


def action_token_totals(records: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "direct_a2a_text_tokens": 0,
        "graph_update_tokens": 0,
        "graph_read_context_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_model_tokens": 0,
        "physical_input_tokens": 0,
        "physical_llm_input_tokens": 0,
        "logical_input_tokens": 0,
        "agent_output_tokens": 0,
        "prefill_cost_tokens": 0,
        "decode_cost_tokens": 0,
        "forward_calls": 0,
        "peak_context_tokens": 0,
        "reasoning_round_count": 0,
        "solver_revision_round_count": 0,
        "critic_need_fix_count": 0,
        "critic_verified_count": 0,
        "final_round_id_sum": 0,
        "max_final_round_id": 0,
        "persistent_context_tokens": 0,
        "incremental_context_tokens": 0,
        "logical_communication_tokens": 0,
        "system_prompt_tokens": 0,
        "prompt_wrapper_tokens": 0,
        "reusable_prefix_tokens": 0,
        "unique_prefix_tokens": 0,
        "repeated_prefix_tokens": 0,
        "simulated_context_reuse_input_tokens": 0,
        "graph_context_content_tokens": 0,
        "graph_context_wrapper_tokens": 0,
        "graph_context_edge_tokens": 0,
        "graph_source_duplicate_in_prompt_tokens": 0,
        "graph_source_duplicate_in_prompt_original_tokens": 0,
        "graph_source_deduplicated_prompt_saved_tokens": 0,
        "graph_source_cross_call_reread_tokens": 0,
        "graph_role_aware_source_ref_saved_tokens": 0,
        "graph_role_aware_state_ref_saved_tokens": 0,
    }
    for record in records:
        runtime = record.get("runtime_summary", {})
        final_round_id = int(record.get("final_round_id", 0) or 0)
        totals["final_round_id_sum"] += final_round_id
        totals["max_final_round_id"] = max(totals["max_final_round_id"], final_round_id)
        totals["direct_a2a_text_tokens"] += int(runtime.get("a2a_text_tokens", 0) or 0)
        seen_reusable_prefix_keys: set[str] = set()
        for action_log in runtime.get("records", []):
            totals["reasoning_round_count"] += 1
            if action_log.get("role") == "solver" and action_log.get("mode") == "repair":
                totals["solver_revision_round_count"] += 1
            action = action_log.get("action", {})
            if action_log.get("role") == "critic" and isinstance(action, dict):
                if action.get("op") == "verify" and action.get("status") == "need_fix":
                    totals["critic_need_fix_count"] += 1
                if action.get("op") == "verify" and action.get("status") == "verified":
                    totals["critic_verified_count"] += 1
            telemetry = action_log.get("telemetry", {}) if isinstance(action_log, dict) else {}
            physical_input = int(telemetry.get("physical_input_tokens", 0) or 0)
            logical_input = int(telemetry.get("logical_input_tokens", action_log.get("logical_context_tokens", 0)) or 0)
            output_tokens = int(telemetry.get("output_tokens", 0) or 0)
            reusable_prefix_tokens = int(telemetry.get("reusable_prefix_tokens", 0) or 0)
            reusable_prefix_key = str(telemetry.get("reusable_prefix_key", ""))
            repeated_prefix_tokens = 0
            if reusable_prefix_key:
                if reusable_prefix_key in seen_reusable_prefix_keys:
                    repeated_prefix_tokens = reusable_prefix_tokens
                    totals["repeated_prefix_tokens"] += repeated_prefix_tokens
                else:
                    seen_reusable_prefix_keys.add(reusable_prefix_key)
                    totals["unique_prefix_tokens"] += reusable_prefix_tokens
            context_tokens = int(
                telemetry.get(
                    "graph_read_context_tokens",
                    action_log.get("context_tokens", action_log.get("logical_context_tokens", 0)),
                )
                or 0
            )
            graph_update = int(telemetry.get("graph_update_tokens", action_log.get("graph_update_tokens", 0)) or 0)
            totals["physical_input_tokens"] += physical_input
            totals["physical_llm_input_tokens"] += int(
                telemetry.get("physical_llm_input_tokens", physical_input) or 0
            )
            totals["logical_input_tokens"] += logical_input
            totals["total_input_tokens"] += physical_input
            totals["total_output_tokens"] += output_tokens
            totals["agent_output_tokens"] += output_tokens
            totals["prefill_cost_tokens"] += int(telemetry.get("prefill_cost_tokens", physical_input) or 0)
            totals["decode_cost_tokens"] += int(telemetry.get("decode_cost_tokens", output_tokens) or 0)
            totals["graph_read_context_tokens"] += context_tokens
            for key in (
                "persistent_context_tokens",
                "incremental_context_tokens",
                "logical_communication_tokens",
                "system_prompt_tokens",
                "prompt_wrapper_tokens",
                "reusable_prefix_tokens",
                "graph_context_content_tokens",
                "graph_context_wrapper_tokens",
                "graph_context_edge_tokens",
                "graph_source_duplicate_in_prompt_tokens",
                "graph_source_duplicate_in_prompt_original_tokens",
                "graph_source_deduplicated_prompt_saved_tokens",
                "graph_source_cross_call_reread_tokens",
                "graph_role_aware_source_ref_saved_tokens",
                "graph_role_aware_state_ref_saved_tokens",
            ):
                totals[key] += int(telemetry.get(key, 0) or 0)
            totals["simulated_context_reuse_input_tokens"] += max(
                0,
                physical_input - min(physical_input, repeated_prefix_tokens),
            )
            totals["graph_update_tokens"] += graph_update
            totals["forward_calls"] += int(telemetry.get("forward_calls", 0) or 0)
            totals["peak_context_tokens"] = max(totals["peak_context_tokens"], context_tokens)
    totals["total_model_tokens"] = totals["total_input_tokens"] + totals["total_output_tokens"]
    totals["avg_final_round_id"] = (
        totals["final_round_id_sum"] / len(records) if records else 0.0
    )
    totals["communication_cost_tokens"] = int(totals.get("total_comm_tokens", 0) or 0)
    totals["semantic_communication_cost_tokens"] = int(totals.get("logical_communication_tokens", 0) or 0)
    totals["inference_prefill_cost_tokens"] = int(totals.get("prefill_cost_tokens", 0) or 0)
    totals["inference_decode_cost_tokens"] = int(totals.get("decode_cost_tokens", 0) or 0)
    totals["phase8_total_cost_tokens"] = (
        totals["communication_cost_tokens"]
        + totals["inference_prefill_cost_tokens"]
        + totals["inference_decode_cost_tokens"]
    )
    totals["context_reuse_saving_ratio"] = (
        int(totals.get("repeated_prefix_tokens", 0) or 0) / totals["total_input_tokens"]
        if totals["total_input_tokens"] else 0.0
    )
    totals["logical_vs_physical_input_gap_tokens"] = (
        totals["total_input_tokens"] - int(totals.get("logical_communication_tokens", 0) or 0)
    )
    return totals


def optional_consumption_totals(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "optional_transmitted_roots": 0,
        "optional_transmitted_root_tokens": 0,
        "optional_consumed_roots": 0,
        "optional_consumed_root_tokens": 0,
        "optional_consumption_rate": 0.0,
        "optimization_headroom": 0.0,
        "oracle_saving_upper_bound": 0.0,
        "optional_consumption_by_pair": {},
    }
    total_context_tokens = 0
    total_model_input_tokens = 0
    by_pair: Dict[str, Dict[str, Any]] = {}

    for record in records:
        logs = record.get("runtime_summary", {}).get("records", [])
        for log in logs:
            total_context_tokens += int(log.get("context_tokens", 0) or 0)
            telemetry = log.get("telemetry", {}) if isinstance(log, dict) else {}
            total_model_input_tokens += int(telemetry.get("physical_input_tokens", 0) or 0)

        for index, log in enumerate(logs):
            for event in log.get("communication", []):
                receiver = str(event.get("receiver", ""))
                pair = f"{event.get('sender', '')}->{receiver}"
                pair_totals = by_pair.setdefault(pair, {
                    "optional_transmitted_roots": 0,
                    "optional_transmitted_root_tokens": 0,
                    "optional_consumed_roots": 0,
                    "optional_consumed_root_tokens": 0,
                    "consumed_node_ids": [],
                    "transmitted_node_ids": [],
                })
                selected = list(event.get("selected_optional_root_node_ids", []))
                token_by_node = event.get("selected_optional_root_token_by_node", {})
                if not isinstance(token_by_node, dict):
                    token_by_node = {}
                future_context_nodes: set[str] = set()
                for future in logs[index + 1:]:
                    if future.get("role") != receiver:
                        continue
                    future_context_nodes.update(str(node_id) for node_id in future.get("context_slice_node_ids", []))
                consumed = [node_id for node_id in selected if node_id in future_context_nodes]
                transmitted_tokens = sum(int(token_by_node.get(node_id, 0) or 0) for node_id in selected)
                consumed_tokens = sum(int(token_by_node.get(node_id, 0) or 0) for node_id in consumed)

                totals["optional_transmitted_roots"] += len(selected)
                totals["optional_transmitted_root_tokens"] += transmitted_tokens
                totals["optional_consumed_roots"] += len(consumed)
                totals["optional_consumed_root_tokens"] += consumed_tokens
                pair_totals["optional_transmitted_roots"] += len(selected)
                pair_totals["optional_transmitted_root_tokens"] += transmitted_tokens
                pair_totals["optional_consumed_roots"] += len(consumed)
                pair_totals["optional_consumed_root_tokens"] += consumed_tokens
                pair_totals["transmitted_node_ids"].extend(selected)
                pair_totals["consumed_node_ids"].extend(consumed)

    if totals["optional_transmitted_roots"]:
        totals["optional_consumption_rate"] = (
            totals["optional_consumed_roots"] / totals["optional_transmitted_roots"]
        )
    if total_context_tokens:
        totals["optimization_headroom"] = totals["optional_consumed_root_tokens"] / total_context_tokens
    if total_model_input_tokens:
        totals["oracle_saving_upper_bound"] = totals["optional_consumed_root_tokens"] / total_model_input_tokens
    for pair_totals in by_pair.values():
        transmitted = int(pair_totals["optional_transmitted_roots"])
        pair_totals["optional_consumption_rate"] = (
            int(pair_totals["optional_consumed_roots"]) / transmitted if transmitted else 0.0
        )
        pair_totals["transmitted_node_ids"] = sorted(set(pair_totals["transmitted_node_ids"]))
        pair_totals["consumed_node_ids"] = sorted(set(pair_totals["consumed_node_ids"]))
    totals["optional_consumption_by_pair"] = by_pair
    return totals


def evaluate_verifier_guided_candidates(
    *,
    rows: List[Dict[str, Any]],
    domain: str,
    model: Any,
    candidate_count: int,
    seed: int,
    temperature: float,
    top_p: float,
    current_baseline: float,
    data_path: str = "",
) -> Dict[str, Any]:
    if domain not in {"gsm8k", "humaneval", "mbpp"}:
        raise ValueError("verifier_guided_candidates currently supports gsm8k, humaneval, and mbpp")
    from scripts.run_reasoning_upper_bounds import run_verifier_select

    report = run_verifier_select(
        model,
        rows,
        domain=domain,
        n=candidate_count,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        current_baseline=current_baseline,
    )
    records = list(report.get("records", []))
    metric_values: Dict[str, float] = {}
    for key in ("coverage@1", "coverage@2", "coverage@4", "coverage@8", "coverage@16"):
        if key in report:
            metric_values[key] = float(report[key])
    return {
        **report,
        "domain": domain,
        "data_path": data_path,
        "execution_mode": "verifier_guided_candidates",
        "candidate_count": candidate_count,
        "count": len(records),
        "accuracy_or_pass_at_1": float(report.get("verifier_select_at_n", 0.0) or 0.0),
        **metric_values,
        "failure_count": 0,
        "candidate_graph_invariant_error_count": sum(
            int(bool(record.get("candidate_graph", {}).get("invariant_errors")))
            for record in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("gsm8k", "tatqa", "hotpotqa", "mbpp", "humaneval", "mmlu_pro"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--execution-mode", choices=("optimized", "native_langgraph", "verifier_guided_candidates"), default="optimized")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--current-baseline", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--communication-policy",
        choices=(
            "send_all",
            "minimal_no_feedback",
            "minimal_sendall_fallback",
            "minimal_targeted_feedback",
            "random_keep_75",
            "random_keep_50",
            "random_keep_25",
            "closure_aware_heuristic",
            "random_same_budget",
            "static_utility",
            "receiver_aware_heuristic",
            "fragment_ablation_core_only",
            "fragment_ablation_support_dependencies",
            "fragment_ablation_full_plan",
            "fragment_ablation_result_metadata",
            "fragment_ablation_calculation_trace",
            "fragment_ablation_full_feedback",
        ),
        default="closure_aware_heuristic",
    )
    parser.add_argument("--communication-seed", type=int, default=0)
    parser.add_argument("--communication-budget-tokens", type=int, default=None)
    parser.add_argument(
        "--graph-context-mode",
        choices=("baseline", "deduplicated", "source_state_split", "role_aware"),
        default="baseline",
    )
    parser.add_argument("--fragment-utility-table", default="")
    parser.add_argument("--task-family", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = read(Path(args.data_path), args.domain, args.limit)
    if not rows:
        raise SystemExit("No evaluation rows loaded")
    if args.execution_mode == "verifier_guided_candidates":
        from scripts.run_reasoning_upper_bounds import SamplingModel

        model = SamplingModel(args.model_path, max_new_tokens=args.max_new_tokens)
        report = evaluate_verifier_guided_candidates(
            rows=rows,
            domain=args.domain,
            model=model,
            candidate_count=args.candidate_count,
            seed=args.communication_seed,
            temperature=args.temperature,
            top_p=args.top_p,
            current_baseline=args.current_baseline,
            data_path=args.data_path,
        )
        report.update({
            "model_path": args.model_path,
            "limit": args.limit,
            "seed": args.communication_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        })
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))
        return

    from langgraph.checkpoint.memory import MemorySaver
    from workflow_runtime.communication import make_communication_policy
    from workflow_runtime.langgraph_workflow import LangGraphWorkflow, NativeLangGraphWorkflow
    from workflow_runtime.model_backend import DirectTransformersModel, TransformersModel
    import main as workflow

    model = (
        DirectTransformersModel(args.model_path)
        if args.execution_mode == "native_langgraph"
        else TransformersModel(args.model_path)
    )
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = f"eval_{args.domain}_{index}"
        task = f"[domain={row['task_type']}]\n{row['question']}"
        graph = None
        if args.execution_mode == "native_langgraph":
            langgraph_workflow = NativeLangGraphWorkflow(
                model=model,
                task_id=sample_id,
                task=render_native_task(row),
                task_type=row["task_type"],
                max_rounds=args.max_rounds,
            )
        else:
            graph = workflow.init_workflow_graph(task, task_id=sample_id, task_type=row["task_type"])
            add_source_nodes(workflow, graph, row, sample_id)
            langgraph_workflow = LangGraphWorkflow(
                store=graph,
                model=model,
                task_id=sample_id,
                task_type=row["task_type"],
                tokenizer=model.tokenizer,
                max_rounds=args.max_rounds,
                communication_policy=make_communication_policy(
                    args.communication_policy,
                    seed=args.communication_seed,
                    utility_table_path=args.fragment_utility_table or None,
                    task_family=args.task_family or args.domain,
                ),
                communication_budget_tokens=args.communication_budget_tokens,
                graph_context_mode=args.graph_context_mode,
            )
        failure = ""
        started = time.time()
        run_result: Dict[str, Any] = {}
        try:
            app = langgraph_workflow.compile(checkpointer=MemorySaver())
            run_result = app.invoke(
                langgraph_workflow.initial_state(),
                {"configurable": {"thread_id": sample_id}},
            )
        except Exception as exc:
            failure = str(exc)
        if failure:
            scores = {"correct": False, "predicted_answer": None}
        elif args.execution_mode == "native_langgraph":
            scores = score_prediction(row, run_result.get("final_answer"))
        else:
            assert graph is not None
            scores = score_record(workflow, row, graph)
        records.append({
            "sample_id": sample_id,
            "domain": args.domain,
            "correct": bool(not failure and scores.get("correct")),
            "scores": scores,
            "gold_answer": row.get("gold_answer"),
            "failure": failure,
            "latency_sec": time.time() - started,
            "runtime_summary": {
                "records": run_result.get("logs", []),
            },
            "final_round_id": int(run_result.get("round_id", 0) or 0),
            "graph_nodes": len(graph.snapshot().nodes) if graph is not None else 0,
            "graph_edges": len(graph.snapshot().edges) if graph is not None else 0,
        })
        print(f"{index + 1}/{len(rows)} correct={records[-1]['correct']} failure={failure or '-'}", flush=True)

    correct = sum(int(item["correct"]) for item in records)
    metric_values: Dict[str, float] = {}
    for key in ("exact_match", "f1", "pass_at_1"):
        values = [float(item["scores"][key]) for item in records if key in item.get("scores", {})]
        if values:
            metric_values[key] = sum(values) / len(values)
    graph_comm = communication_totals(records)
    token_totals = action_token_totals(records)
    optional_consumption = optional_consumption_totals(records)
    phase8_costs = {
        "communication_cost_tokens": int(graph_comm.get("communication_cost_tokens", 0) or 0),
        "semantic_communication_cost_tokens": int(token_totals.get("logical_communication_tokens", 0) or 0),
        "inference_prefill_cost_tokens": int(token_totals.get("prefill_cost_tokens", 0) or 0),
        "inference_decode_cost_tokens": int(token_totals.get("decode_cost_tokens", 0) or 0),
    }
    phase8_costs["phase8_total_cost_tokens"] = (
        phase8_costs["communication_cost_tokens"]
        + phase8_costs["inference_prefill_cost_tokens"]
        + phase8_costs["inference_decode_cost_tokens"]
    )
    report = {
        "domain": args.domain,
        "data_path": args.data_path,
        "execution_mode": args.execution_mode,
        "graph_context_mode": args.graph_context_mode,
        "count": len(records),
        "accuracy_or_pass_at_1": correct / len(records),
        **metric_values,
        "failure_count": sum(bool(item["failure"]) for item in records),
        **token_totals,
        **graph_comm,
        **phase8_costs,
        **optional_consumption,
        "records": records,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
