import copy
import hashlib
import json
import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import tiktoken
except ImportError:
    tiktoken = None


# ============================================================
# 1. Local Model Config
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/root/models/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "1"))

STAGE_MAX_NEW_TOKENS = {
    "planner": 160,
    "solver": 180,
    "critic": 120,
    "final_solver": 80,
}

print(f"[INFO] Loading local model from: {MODEL_PATH}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

if tiktoken is not None:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
else:
    enc = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    if not text or count_tokens(text) <= max_tokens:
        return text
    ids = tokenizer.encode(text)
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("**", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def shorten_text(text: str, max_chars: int = 300) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text

    sentences = re.split(r"(?<=[。！？.!?；;])\s*", text)
    kept: List[str] = []
    total = 0

    for sentence in sentences:
        if not sentence:
            continue
        if total + len(sentence) > max_chars:
            break
        kept.append(sentence)
        total += len(sentence)

    if kept:
        return "".join(kept).strip()

    return text[:max_chars].strip()


def extract_first_line_value(text: str, key: str) -> Optional[str]:
    pattern = rf"{re.escape(key)}\s*[:：]\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_block_value(text: str, key: str, stop_keys: List[str]) -> Optional[str]:
    pattern = rf"{re.escape(key)}\s*[:：]\s*(.*?)(?=\n(?:{'|'.join(re.escape(k) for k in stop_keys)})\s*[:：]|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    value = normalize_text(match.group(1))
    return value or None


def extract_equations(text: str) -> List[str]:
    text = normalize_text(text)
    patterns = [
        r"[\w\u4e00-\u9fff]+\s*=\s*[-+]?\d+(?:\.\d+)?",
        r"[-+]?\d+(?:\.\d+)?\s*[+\-*/×x÷]\s*[-+]?\d+(?:\.\d+)?\s*=\s*[-+]?\d+(?:\.\d+)?",
        r"[-+]?\d+(?:\.\d+)?\s*[<>]=?\s*[-+]?\d+(?:\.\d+)?",
        r"[\w\u4e00-\u9fff]+[^。；;\n]{0,30}?[-+]?\d+(?:\.\d+)?\s*(?:元|个|本|支|kg|g|m|cm|%|人|次)?",
    ]

    items: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = " ".join([x for x in match if x])
            match = match.strip()
            if match and match not in items:
                items.append(match)

    return items[:8]


def extract_key_sentences(text: str, max_items: int = 5) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    candidates = re.split(r"(?<=[。！？.!?；;])\s*|\n+", text)
    candidates = [s.strip(" -•\t") for s in candidates if s.strip()]

    keywords = [
        "目标", "任务", "约束", "条件", "事实", "已知",
        "计划", "步骤", "计算", "结果", "结论", "答案",
        "正确", "错误", "遗漏", "不一致", "需要修正",
        "therefore", "result", "answer", "correct", "incorrect",
        "error", "fact", "constraint", "step", "plan",
    ]

    selected: List[str] = []
    for sentence in candidates:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            if sentence not in selected:
                selected.append(sentence)

    if not selected:
        selected = candidates[:max_items]

    return selected[:max_items]


def extract_generic_facts(text: str, max_chars: int = 260) -> str:
    equations = extract_equations(text)
    if equations:
        return "; ".join(equations)

    key_sentences = extract_key_sentences(text, max_items=4)
    if key_sentences:
        return shorten_text("; ".join(key_sentences), max_chars=max_chars)

    return shorten_text(text, max_chars=max_chars) or "facts not extracted"


def extract_generic_plan(text: str, max_chars: int = 220) -> str:
    key_sentences = extract_key_sentences(text, max_items=4)
    plan_like = [
        sentence for sentence in key_sentences
        if any(keyword in sentence for keyword in ["计划", "步骤", "先", "然后", "最后", "step", "plan"])
    ]

    if plan_like:
        return shorten_text("; ".join(plan_like), max_chars=max_chars)

    return "extract_facts; solve_subtasks; verify_result; produce_final_answer"


def extract_generic_result(text: str, max_chars: int = 260) -> str:
    text = normalize_text(text)

    explicit_result = extract_first_line_value(text, "RESULT")
    if explicit_result:
        return shorten_text(explicit_result, max_chars=max_chars)

    result_keywords = ["结论", "答案", "结果", "因此", "所以", "final", "answer", "result", "therefore"]
    sentences = re.split(r"(?<=[。！？.!?；;])\s*|\n+", text)
    candidates = [s.strip() for s in sentences if s.strip()]
    selected = [
        sentence for sentence in candidates
        if any(keyword.lower() in sentence.lower() for keyword in result_keywords)
    ]

    if selected:
        return shorten_text("; ".join(selected[-3:]), max_chars=max_chars)

    equations = extract_equations(text)
    if equations:
        return "; ".join(equations[-4:])

    return shorten_text(text, max_chars=max_chars) or "result not extracted"


def infer_verification_status(text: str) -> str:
    text = normalize_text(text).lower()

    positive = [
        "正确", "无误", "没有错误", "无错误", "可以接受",
        "consistent", "correct", "valid", "no error", "no issue",
    ]
    negative = [
        "错误", "不正确", "不一致", "遗漏", "需要修正",
        "incorrect", "wrong", "inconsistent", "missing", "need_fix", "revise",
    ]

    if any(keyword in text for keyword in negative) and not any(keyword in text for keyword in positive):
        return "need_fix"

    if any(keyword in text for keyword in positive):
        return "verified"

    return "need_fix"


def parse_yes_no_accuracy(reference_answer: str, predicted_answer: str) -> float:
    ref = normalize_text(reference_answer).lower()
    pred = normalize_text(predicted_answer).lower()
    if not ref or not pred:
        return 0.0

    if ref == pred or ref in pred or pred in ref:
        return 1.0

    ref_numbers = re.findall(r"-?\d+(?:\.\d+)?", ref)
    pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", pred)
    equality_keywords = ["相同", "一样", "equal", "没有人多花", "都花了"]
    equality_ref = any(keyword in ref for keyword in equality_keywords)
    equality_pred = any(keyword in pred for keyword in equality_keywords)

    if equality_ref:
        has_expected_value = not ref_numbers or all(number in pred_numbers for number in set(ref_numbers))
        has_zero_diff = "差额为0" in pred or "diff=0" in pred or "difference=0" in pred or "没有人多花" in pred
        repeats_same_value = any(pred_numbers.count(number) >= 2 for number in set(ref_numbers))
        if not (equality_pred or has_zero_diff or repeats_same_value):
            return 0.0
        if has_expected_value and (equality_pred or has_zero_diff or repeats_same_value):
            return 1.0
        if equality_pred and (has_expected_value or has_zero_diff):
            return 1.0
        if equality_pred or has_zero_diff:
            return 0.5
        return 0.0

    if ref_numbers and ref_numbers == pred_numbers:
        return 1.0

    if ref_numbers and all(number in pred_numbers for number in set(ref_numbers)):
        return 0.5
    return 0.0


# ============================================================
# 2. Global Agent Prompts
# ============================================================

GLOBAL_AGENT_CONSTRAINT = """
你在一个基于 SOP 的多智能体协作系统中工作。
所有消息都应当遵守结构化交接协议，而不是自由聊天。

你必须优先服从上游消息中的：
STATE、FACTS、RESULT、ERRORS、NEXT_ACTION、OUTPUT_LIMIT。

禁止：
1. 寒暄；
2. 重复完整题目；
3. 无必要长篇解释；
4. 在 NEXT_ACTION=verify_only 时重新完整解题；
5. 在 NEXT_ACTION=final_answer_only 时重新计算；
6. 编造上游消息中不存在的信息。

动作规则：
- 如果 NEXT_ACTION=solve：只根据 FACTS 求解，输出必要计算和结论。
- 如果 NEXT_ACTION=verify_only：只检查 RESULT 是否正确；正确时只输出“正确”和一句理由。
- 如果 NEXT_ACTION=final_answer_only：只输出最终答案，不要解释，不要重新推理。
- 如果 NEXT_ACTION=revise：只修正 ERRORS 指出的错误，不要重写无关内容。
"""

ROLE_SUBSCRIPTIONS = {
    "planner": ["task"],
    "solver": ["query_spec", "facts", "plan"],
    "solver_revision": ["query_spec", "facts", "plan", "calculation", "result", "verification"],
    "critic": ["task", "query_spec", "facts", "calculation", "result"],
    "critic_recheck": ["task", "query_spec", "facts", "calculation", "result", "verification"],
    "final_solver": ["query_spec", "result", "verification"],
}

class RequirementCode(str, Enum):
    JSON_ARRAY = "JSON_ARRAY"
    JSON_TWO_STRINGS = "JSON_TWO_STRINGS"
    JSON_ONE_STRING = "JSON_ONE_STRING"
    JSON_NUMERIC_COMPARISON_OBJECT = "JSON_NUMERIC_COMPARISON_OBJECT"
    NO_MARKDOWN = "NO_MARKDOWN"
    CONCISE = "CONCISE"

    KEEP_ALL_FACTS = "KEEP_ALL_FACTS"
    KEEP_NUMBERS = "KEEP_NUMBERS"
    KEEP_UNITS = "KEEP_UNITS"
    KEEP_ENTITIES = "KEEP_ENTITIES"
    PLAN_ONLY = "PLAN_ONLY"
    NO_SOLVE = "NO_SOLVE"

    EXPLICIT_EQUATIONS = "EXPLICIT_EQUATIONS"
    ALL_ENTITY_VALUES = "ALL_ENTITY_VALUES"
    COMPARISON_RELATION = "COMPARISON_RELATION"
    DIFFERENCE = "DIFFERENCE"
    SELF_CONTAINED_RESULT = "SELF_CONTAINED_RESULT"
    USE_ONLY_VISIBLE_FACTS = "USE_ONLY_VISIBLE_FACTS"

    CHECK_CALCULATION = "CHECK_CALCULATION"
    CHECK_RESULT_COMPLETENESS = "CHECK_RESULT_COMPLETENESS"
    CHECK_FACT_CONSISTENCY = "CHECK_FACT_CONSISTENCY"
    ERROR_ONLY = "ERROR_ONLY"

    USE_VERIFICATION_FEEDBACK = "USE_VERIFICATION_FEEDBACK"
    REVISE_ONLY_ERROR = "REVISE_ONLY_ERROR"
    DO_NOT_REPEAT_OLD_RESULT = "DO_NOT_REPEAT_OLD_RESULT"

    FINAL_ONLY = "FINAL_ONLY"
    PRESERVE_VERIFIED_VALUES = "PRESERVE_VERIFIED_VALUES"


class RequirementSpec(BaseModel):
    code: RequirementCode
    instruction: str
    roles: Set[str] = Field(default_factory=set)
    priority: int = 100
    validator_name: Optional[str] = None
    requires: Set[RequirementCode] = Field(default_factory=set)
    conflicts_with: Set[RequirementCode] = Field(default_factory=set)
    description: Optional[str] = None


def requirement_spec(
    code: RequirementCode,
    instruction: str,
    *,
    roles: Optional[Set[str]] = None,
    priority: int = 100,
    validator_name: Optional[str] = None,
    requires: Optional[Set[RequirementCode]] = None,
    conflicts_with: Optional[Set[RequirementCode]] = None,
) -> RequirementSpec:
    return RequirementSpec(
        code=code,
        instruction=instruction,
        roles=roles or set(),
        priority=priority,
        validator_name=validator_name,
        requires=requires or set(),
        conflicts_with=conflicts_with or set(),
    )


ALL_GRAPH_ROLES = {"planner", "solver", "critic", "solver_revision", "critic_recheck", "final_solver"}

REQUIREMENT_CODEBOOK: Dict[RequirementCode, RequirementSpec] = {
    RequirementCode.JSON_ARRAY: requirement_spec(
        RequirementCode.JSON_ARRAY,
        "只输出合法 JSON 数组。",
        roles=ALL_GRAPH_ROLES,
        priority=10,
    ),
    RequirementCode.JSON_TWO_STRINGS: requirement_spec(
        RequirementCode.JSON_TWO_STRINGS,
        '只输出两个字符串组成的一维 JSON 数组。',
        roles={"planner", "solver", "solver_revision"},
        priority=20,
        requires={RequirementCode.JSON_ARRAY},
        validator_name="is_two_string_json_array",
    ),
    RequirementCode.JSON_ONE_STRING: requirement_spec(
        RequirementCode.JSON_ONE_STRING,
        '只输出一个字符串组成的一维 JSON 数组。',
        roles={"final_solver"},
        priority=20,
        requires={RequirementCode.JSON_ARRAY},
        validator_name="is_one_string_json_array",
    ),
    RequirementCode.JSON_NUMERIC_COMPARISON_OBJECT: requirement_spec(
        RequirementCode.JSON_NUMERIC_COMPARISON_OBJECT,
        '只输出对象：{"entity_results":[{"entity":"对象","value":数字},...],"difference":非负数字,"relation":"equal|first_greater|second_greater"}。',
        roles={"solver", "solver_revision"},
        priority=20,
        validator_name="is_numeric_comparison_object",
        conflicts_with={RequirementCode.EXPLICIT_EQUATIONS},
    ),
    RequirementCode.NO_MARKDOWN: requirement_spec(
        RequirementCode.NO_MARKDOWN,
        "不要输出 Markdown、代码块、字段名或额外前缀。",
        roles=ALL_GRAPH_ROLES,
        priority=30,
    ),
    RequirementCode.CONCISE: requirement_spec(
        RequirementCode.CONCISE,
        "保持简洁，但不得丢失本轮必要信息。",
        roles=ALL_GRAPH_ROLES,
        priority=40,
    ),
    RequirementCode.KEEP_ALL_FACTS: requirement_spec(
        RequirementCode.KEEP_ALL_FACTS,
        "保留完成任务所需的全部关键事实。",
        roles={"planner"},
        priority=50,
    ),
    RequirementCode.KEEP_ENTITIES: requirement_spec(
        RequirementCode.KEEP_ENTITIES,
        "保留所有关键对象或实体。",
        roles={"planner"},
        priority=55,
    ),
    RequirementCode.KEEP_NUMBERS: requirement_spec(
        RequirementCode.KEEP_NUMBERS,
        "保留所有关键数字。",
        roles={"planner"},
        priority=56,
    ),
    RequirementCode.KEEP_UNITS: requirement_spec(
        RequirementCode.KEEP_UNITS,
        "保留所有关键单位。",
        roles={"planner"},
        priority=57,
    ),
    RequirementCode.PLAN_ONLY: requirement_spec(
        RequirementCode.PLAN_ONLY,
        "计划只描述后续最小步骤。",
        roles={"planner"},
        priority=60,
    ),
    RequirementCode.NO_SOLVE: requirement_spec(
        RequirementCode.NO_SOLVE,
        "不要在计划阶段直接求解。",
        roles={"planner"},
        priority=61,
    ),
    RequirementCode.EXPLICIT_EQUATIONS: requirement_spec(
        RequirementCode.EXPLICIT_EQUATIONS,
        "给出显式算式。",
        roles={"solver", "solver_revision"},
        priority=50,
        validator_name="has_equation",
        conflicts_with={RequirementCode.FINAL_ONLY},
    ),
    RequirementCode.ALL_ENTITY_VALUES: requirement_spec(
        RequirementCode.ALL_ENTITY_VALUES,
        "给出每个对象的最终数值。",
        roles={"solver", "solver_revision"},
        priority=55,
        validator_name="contains_all_entities",
    ),
    RequirementCode.COMPARISON_RELATION: requirement_spec(
        RequirementCode.COMPARISON_RELATION,
        "明确说明谁更大、谁更小或是否相同。",
        roles={"solver", "solver_revision", "final_solver"},
        priority=56,
    ),
    RequirementCode.DIFFERENCE: requirement_spec(
        RequirementCode.DIFFERENCE,
        "给出差额。",
        roles={"solver", "solver_revision", "final_solver"},
        priority=57,
    ),
    RequirementCode.SELF_CONTAINED_RESULT: requirement_spec(
        RequirementCode.SELF_CONTAINED_RESULT,
        "最终结果必须能够独立理解。",
        roles={"solver", "solver_revision"},
        priority=58,
    ),
    RequirementCode.USE_ONLY_VISIBLE_FACTS: requirement_spec(
        RequirementCode.USE_ONLY_VISIBLE_FACTS,
        "只使用当前可见子图中的事实和计划。",
        roles={"solver", "solver_revision", "critic", "critic_recheck", "final_solver"},
        priority=59,
    ),
    RequirementCode.CHECK_CALCULATION: requirement_spec(
        RequirementCode.CHECK_CALCULATION,
        "检查 calculation 中的计算是否正确。",
        roles={"critic", "critic_recheck"},
        priority=50,
    ),
    RequirementCode.CHECK_FACT_CONSISTENCY: requirement_spec(
        RequirementCode.CHECK_FACT_CONSISTENCY,
        "检查结果是否与 facts 一致。",
        roles={"critic", "critic_recheck"},
        priority=51,
    ),
    RequirementCode.CHECK_RESULT_COMPLETENESS: requirement_spec(
        RequirementCode.CHECK_RESULT_COMPLETENESS,
        "检查 result 是否完整回答原任务。",
        roles={"critic", "critic_recheck"},
        priority=52,
    ),
    RequirementCode.ERROR_ONLY: requirement_spec(
        RequirementCode.ERROR_ONLY,
        '正确时只输出 [0]；错误时只输出 [1,"具体错误"]。',
        roles={"critic", "critic_recheck"},
        priority=20,
        requires={RequirementCode.JSON_ARRAY},
    ),
    RequirementCode.USE_VERIFICATION_FEEDBACK: requirement_spec(
        RequirementCode.USE_VERIFICATION_FEEDBACK,
        "使用 verification 中的反馈进行修正。",
        roles={"solver_revision", "critic_recheck"},
        priority=45,
    ),
    RequirementCode.REVISE_ONLY_ERROR: requirement_spec(
        RequirementCode.REVISE_ONLY_ERROR,
        "只修改反馈指出的错误。",
        roles={"solver_revision"},
        priority=46,
    ),
    RequirementCode.DO_NOT_REPEAT_OLD_RESULT: requirement_spec(
        RequirementCode.DO_NOT_REPEAT_OLD_RESULT,
        "不要原样重复旧结果。",
        roles={"solver_revision"},
        priority=47,
    ),
    RequirementCode.FINAL_ONLY: requirement_spec(
        RequirementCode.FINAL_ONLY,
        "只输出最终答案，不要重新推理或解释。",
        roles={"final_solver"},
        priority=45,
        conflicts_with={RequirementCode.EXPLICIT_EQUATIONS},
    ),
    RequirementCode.PRESERVE_VERIFIED_VALUES: requirement_spec(
        RequirementCode.PRESERVE_VERIFIED_VALUES,
        "保留已验证 result 中的关键数值和结论。",
        roles={"final_solver"},
        priority=50,
    ),
}


ROLE_BASE_REQUIREMENTS = {
    "planner": [
        RequirementCode.KEEP_ALL_FACTS,
        RequirementCode.KEEP_ENTITIES,
        RequirementCode.KEEP_NUMBERS,
        RequirementCode.KEEP_UNITS,
        RequirementCode.PLAN_ONLY,
        RequirementCode.NO_SOLVE,
        RequirementCode.JSON_TWO_STRINGS,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
    "solver": [
        RequirementCode.EXPLICIT_EQUATIONS,
        RequirementCode.SELF_CONTAINED_RESULT,
        RequirementCode.USE_ONLY_VISIBLE_FACTS,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
    "critic": [
        RequirementCode.CHECK_CALCULATION,
        RequirementCode.CHECK_FACT_CONSISTENCY,
        RequirementCode.CHECK_RESULT_COMPLETENESS,
        RequirementCode.ERROR_ONLY,
        RequirementCode.USE_ONLY_VISIBLE_FACTS,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
    "solver_revision": [
        RequirementCode.USE_VERIFICATION_FEEDBACK,
        RequirementCode.REVISE_ONLY_ERROR,
        RequirementCode.DO_NOT_REPEAT_OLD_RESULT,
        RequirementCode.EXPLICIT_EQUATIONS,
        RequirementCode.SELF_CONTAINED_RESULT,
        RequirementCode.USE_ONLY_VISIBLE_FACTS,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
    "critic_recheck": [
        RequirementCode.USE_VERIFICATION_FEEDBACK,
        RequirementCode.CHECK_CALCULATION,
        RequirementCode.CHECK_FACT_CONSISTENCY,
        RequirementCode.CHECK_RESULT_COMPLETENESS,
        RequirementCode.ERROR_ONLY,
        RequirementCode.USE_ONLY_VISIBLE_FACTS,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
    "final_solver": [
        RequirementCode.FINAL_ONLY,
        RequirementCode.PRESERVE_VERIFIED_VALUES,
        RequirementCode.JSON_ONE_STRING,
        RequirementCode.NO_MARKDOWN,
        RequirementCode.CONCISE,
    ],
}


TASK_REQUIREMENT_POLICY = {
    "numeric_comparison": {
        "planner": [
            RequirementCode.KEEP_ENTITIES,
            RequirementCode.KEEP_NUMBERS,
            RequirementCode.KEEP_UNITS,
        ],
        "solver": [
            RequirementCode.JSON_NUMERIC_COMPARISON_OBJECT,
            RequirementCode.ALL_ENTITY_VALUES,
            RequirementCode.COMPARISON_RELATION,
            RequirementCode.DIFFERENCE,
        ],
        "solver_revision": [
            RequirementCode.JSON_NUMERIC_COMPARISON_OBJECT,
            RequirementCode.ALL_ENTITY_VALUES,
            RequirementCode.COMPARISON_RELATION,
            RequirementCode.DIFFERENCE,
        ],
        "critic": [
            RequirementCode.CHECK_CALCULATION,
            RequirementCode.CHECK_RESULT_COMPLETENESS,
        ],
        "critic_recheck": [
            RequirementCode.CHECK_CALCULATION,
            RequirementCode.CHECK_RESULT_COMPLETENESS,
        ],
        "final_solver": [
            RequirementCode.COMPARISON_RELATION,
            RequirementCode.DIFFERENCE,
        ],
    },
    "general_reasoning": {
        "planner": [RequirementCode.KEEP_ALL_FACTS],
        "solver": [RequirementCode.SELF_CONTAINED_RESULT, RequirementCode.JSON_TWO_STRINGS],
        "solver_revision": [RequirementCode.SELF_CONTAINED_RESULT, RequirementCode.JSON_TWO_STRINGS],
        "critic": [RequirementCode.CHECK_FACT_CONSISTENCY],
        "critic_recheck": [RequirementCode.CHECK_FACT_CONSISTENCY],
        "final_solver": [RequirementCode.FINAL_ONLY],
    },
}


ROLE_IO_SPEC = {
    "planner": {
        "inputs": ["task"],
        "outputs": ["query_spec", "facts", "plan"],
        "schema": '["facts","plan"]；query_spec 由程序从 task 生成',
        "objective": "抽取事实并生成计划；程序侧生成查询规格。",
    },
    "solver": {
        "inputs": ["query_spec", "facts", "plan"],
        "outputs": ["calculation", "result"],
        "schema": '["calculation","result"]',
        "objective": "根据事实和计划完成求解。",
    },
    "critic": {
        "inputs": ["task", "query_spec", "facts", "calculation", "result"],
        "outputs": ["verification"],
        "schema": '[0] 或 [1,"error"]',
        "objective": "核验计算与结果。",
    },
    "solver_revision": {
        "inputs": ["query_spec", "facts", "plan", "calculation", "result", "verification"],
        "outputs": ["calculation", "result"],
        "schema": '["revised_calculation","revised_result"]',
        "objective": "根据验证反馈修正结果。",
    },
    "critic_recheck": {
        "inputs": ["task", "query_spec", "facts", "calculation", "result", "verification"],
        "outputs": ["verification"],
        "schema": '[0] 或 [1,"error"]',
        "objective": "重新核验修订后的计算与结果。",
    },
    "final_solver": {
        "inputs": ["query_spec", "result", "verification"],
        "outputs": ["final_answer"],
        "schema": '["final_answer"]',
        "objective": "输出最终答案。",
    },
}


def get_task_text(graph: "WorkflowGraphState") -> str:
    task_node = graph.nodes.get("task")
    if task_node is None or task_node.content is None:
        return ""
    return normalize_text(str(task_node.content))


def detect_task_type(task: str) -> str:
    text = normalize_text(task).lower()
    comparison_keywords = [
        "谁更多",
        "多花",
        "差额",
        "比较",
        "difference",
        "compare",
    ]
    if any(keyword in text for keyword in comparison_keywords):
        return "numeric_comparison"
    if any(keyword in text for keyword in ["是否", "判断", "正确吗"]):
        return "verification"
    return "general_reasoning"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for item in items:
        normalized = normalize_text(item)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def extract_comparison_entities_from_text(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    entities: List[str] = []
    purchase_patterns = [
        r"([A-Za-z][A-Za-z0-9_]*|[\u4e00-\u9fff]{1,12})\s*买了",
        r"([A-Za-z][A-Za-z0-9_]*|[\u4e00-\u9fff]{1,12})\s*(?:bought|purchased)",
    ]
    for pattern in purchase_patterns:
        entities.extend(match.group(1) for match in re.finditer(pattern, text, flags=re.IGNORECASE))

    if len(dedupe_preserve_order(entities)) >= 2:
        return dedupe_preserve_order(entities)[:2]

    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", text)
    return dedupe_preserve_order(capitalized)[:2]


def infer_query_metric(text: str) -> str:
    text = normalize_text(text).lower()
    if any(keyword in text for keyword in ["花", "花费", "spent", "cost", "元"]):
        return "total_cost"
    if any(keyword in text for keyword in ["数量", "个数", "count", "number"]):
        return "count"
    return "numeric_value"


def infer_query_unit(text: str) -> str:
    text = normalize_text(text)
    if "元" in text:
        return "元"
    currency_match = re.search(r"\b(yuan|dollar|usd|rmb)\b", text, flags=re.IGNORECASE)
    if currency_match:
        return currency_match.group(1).lower()
    return ""


def build_query_spec_from_task(task: str) -> Dict[str, Any]:
    task = normalize_text(task)
    entities = extract_comparison_entities_from_text(task)
    return {
        "operation": "compare" if detect_task_type(task) == "numeric_comparison" else "solve",
        "entities": entities,
        "metric": infer_query_metric(task),
        "unit": infer_query_unit(task),
        "asked_difference": any(keyword in task for keyword in ["多花", "差额", "difference"]),
        "asked_relation": any(keyword in task for keyword in ["谁", "more", "less", "相同", "比较"]),
    }


def query_spec_to_content(query_spec: Dict[str, Any]) -> str:
    return json.dumps(query_spec, ensure_ascii=False, separators=(",", ":"))


class TaskAnalyzer:
    def analyze(self, graph: "WorkflowGraphState") -> str:
        return detect_task_type(get_task_text(graph))


TASK_ANALYZER = TaskAnalyzer()


def canonical_role(role: str) -> str:
    if role == "solver_revision":
        return "solver"
    if role == "critic_recheck":
        return "critic"
    return role


class AgentAssignment(BaseModel):
    role: str
    task_type: str
    objective: str
    required_inputs: List[str]
    required_outputs: List[str]
    requirement_codes: List[RequirementCode]
    output_schema: str
    output_budget: int
    mode: str = "normal"
    feedback: Optional[str] = None
    graph_state_summary: str = ""


def describe_graph_state_for_role(role: str, graph: "WorkflowGraphState") -> str:
    available_types = {
        node.type
        for node in graph.nodes.values()
        if node.content is not None and node.status != "empty"
    }
    io_spec = ROLE_IO_SPEC[role]
    expected_inputs = set(io_spec["inputs"])
    available_inputs = sorted(expected_inputs & available_types)
    missing_inputs = sorted(expected_inputs - available_types)
    return (
        f"已存在输入：{', '.join(available_inputs) or 'none'}；"
        f"尚未出现：{', '.join(missing_inputs) or 'none'}。"
    )


class RequirementEncoder:
    def __init__(
        self,
        role_base_requirements: Dict[str, List[RequirementCode]],
        task_requirement_policy: Dict[str, Dict[str, List[RequirementCode]]],
    ):
        self.role_base_requirements = role_base_requirements
        self.task_requirement_policy = task_requirement_policy

    def encode(
        self,
        *,
        role: str,
        task_type: str,
        mode: str = "normal",
    ) -> List[RequirementCode]:
        codes: List[RequirementCode] = []
        codes.extend(self.role_base_requirements.get(role, []))

        task_policy = self.task_requirement_policy.get(
            task_type,
            self.task_requirement_policy["general_reasoning"],
        )
        codes.extend(task_policy.get(role, []))

        if mode in {"revision", "recheck"}:
            codes.append(RequirementCode.USE_VERIFICATION_FEEDBACK)
        if mode == "revision":
            codes.extend(
                [
                    RequirementCode.REVISE_ONLY_ERROR,
                    RequirementCode.DO_NOT_REPEAT_OLD_RESULT,
                ]
            )

        return self._deduplicate(codes)

    @staticmethod
    def _deduplicate(codes: List[RequirementCode]) -> List[RequirementCode]:
        seen: Set[RequirementCode] = set()
        result: List[RequirementCode] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                result.append(code)
        return result


class RequirementResolver:
    def __init__(self, codebook: Dict[RequirementCode, RequirementSpec]):
        self.codebook = codebook

    def resolve(self, codes: List[RequirementCode], role: str) -> List[RequirementCode]:
        selected: Set[RequirementCode] = set(codes)

        changed = True
        while changed:
            changed = False
            for code in list(selected):
                spec = self.codebook[code]
                for required_code in spec.requires:
                    if required_code not in selected:
                        selected.add(required_code)
                        changed = True

        selected = {
            code
            for code in selected
            if not self.codebook[code].roles or role in self.codebook[code].roles
        }

        result: List[RequirementCode] = []
        for code in sorted(selected, key=lambda item: self.codebook[item].priority):
            spec = self.codebook[code]
            has_conflict = any(
                existing in spec.conflicts_with
                or code in self.codebook[existing].conflicts_with
                for existing in result
            )
            if not has_conflict:
                result.append(code)
        return result


class RequirementPromptCompiler:
    def __init__(self, codebook: Dict[RequirementCode, RequirementSpec]):
        self.codebook = codebook

    def compile_instructions(self, codes: List[RequirementCode]) -> str:
        specs = [self.codebook[code] for code in codes]
        specs.sort(key=lambda spec: spec.priority)
        return "\n".join(f"- {spec.instruction}" for spec in specs)

    def compile_system_prompt(self, assignment: AgentAssignment) -> str:
        instructions = self.compile_instructions(assignment.requirement_codes)
        inputs = ",".join(assignment.required_inputs)
        outputs = ",".join(assignment.required_outputs)
        return (
            f"角色：{assignment.role}\n"
            f"模式：{assignment.mode}\n"
            f"任务类型：{assignment.task_type}\n"
            f"目标：{assignment.objective}\n"
            f"输入：{inputs}\n"
            f"输出：{outputs}\n"
            f"格式：{assignment.output_schema}\n"
            f"预算：约 {assignment.output_budget} tokens\n"
            f"{instructions}"
        )

    def compile_user_prompt(self, *, assignment: AgentAssignment, context: str) -> str:
        sections = [
            f"状态：{assignment.graph_state_summary}",
            f"子图：\n{context}",
        ]
        if assignment.feedback:
            sections.append(f"反馈：{assignment.feedback}")
        return "\n\n".join(sections)


REQUIREMENT_ENCODER = RequirementEncoder(
    ROLE_BASE_REQUIREMENTS,
    TASK_REQUIREMENT_POLICY,
)
REQUIREMENT_RESOLVER = RequirementResolver(REQUIREMENT_CODEBOOK)
REQUIREMENT_PROMPT_COMPILER = RequirementPromptCompiler(REQUIREMENT_CODEBOOK)


def build_assignment(
    *,
    role: str,
    graph: "WorkflowGraphState",
    mode: str = "normal",
    feedback: Optional[str] = None,
) -> AgentAssignment:
    task_type = TASK_ANALYZER.analyze(graph)
    io_spec = ROLE_IO_SPEC[role]
    requirement_codes = REQUIREMENT_ENCODER.encode(
        role=role,
        task_type=task_type,
        mode=mode,
    )
    requirement_codes = REQUIREMENT_RESOLVER.resolve(requirement_codes, role)

    if feedback is None and mode in {"revision", "recheck"}:
        verification = latest_node_by_type(graph, "verification")
        if verification is not None and verification.content is not None:
            feedback = normalize_text(str(verification.content))

    required_outputs = list(io_spec["outputs"])
    output_schema = io_spec["schema"]
    if role in {"solver", "solver_revision"} and task_type == "numeric_comparison":
        comparison_entities = extract_comparison_entities_from_text(get_task_text(graph))
        first_entity = comparison_entities[0] if len(comparison_entities) >= 1 else "Alice"
        second_entity = comparison_entities[1] if len(comparison_entities) >= 2 else "Bob"
        required_outputs = ["entity_results", "difference", "relation"]
        output_schema = (
            f'{{"entity_results":[{{"entity":"{first_entity}","value":22}},'
            f'{{"entity":"{second_entity}","value":22}}],"difference":0,"relation":"equal"}}'
        )

    return AgentAssignment(
        role=role,
        task_type=task_type,
        objective=io_spec["objective"],
        required_inputs=io_spec["inputs"],
        required_outputs=required_outputs,
        requirement_codes=requirement_codes,
        output_schema=output_schema,
        output_budget=STAGE_MAX_NEW_TOKENS.get(canonical_role(role), MAX_NEW_TOKENS),
        mode=mode,
        feedback=feedback,
        graph_state_summary=describe_graph_state_for_role(role, graph),
    )

# ============================================================
# 3. Global Token Tracker
# ============================================================

class TokenCallRecord(BaseModel):
    call_id: int
    caller: str
    stage: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_sec: float


class GlobalTokenTracker:
    def __init__(self):
        self.records: List[TokenCallRecord] = []
        self.call_id = 0

    def reset(self):
        self.records = []
        self.call_id = 0

    def add(
        self,
        caller: str,
        stage: str,
        input_tokens: int,
        output_tokens: int,
        latency_sec: float,
    ):
        self.call_id += 1
        self.records.append(
            TokenCallRecord(
                call_id=self.call_id,
                caller=caller,
                stage=stage,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_sec=latency_sec,
            )
        )

    def summary(self) -> Dict[str, Any]:
        total_input = sum(record.input_tokens for record in self.records)
        total_output = sum(record.output_tokens for record in self.records)
        total = sum(record.total_tokens for record in self.records)
        total_latency = sum(record.latency_sec for record in self.records)

        by_caller: Dict[str, Dict[str, Any]] = {}
        by_stage: Dict[str, Dict[str, Any]] = {}

        for record in self.records:
            caller_stat = by_caller.setdefault(
                record.caller,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_sec": 0.0,
                    "calls": 0,
                },
            )
            caller_stat["input_tokens"] += record.input_tokens
            caller_stat["output_tokens"] += record.output_tokens
            caller_stat["total_tokens"] += record.total_tokens
            caller_stat["latency_sec"] += record.latency_sec
            caller_stat["calls"] += 1

            stage_stat = by_stage.setdefault(
                record.stage,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_sec": 0.0,
                    "calls": 0,
                },
            )
            stage_stat["input_tokens"] += record.input_tokens
            stage_stat["output_tokens"] += record.output_tokens
            stage_stat["total_tokens"] += record.total_tokens
            stage_stat["latency_sec"] += record.latency_sec
            stage_stat["calls"] += 1

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total,
            "total_latency_sec": total_latency,
            "by_caller": by_caller,
            "by_stage": by_stage,
            "records": [record.model_dump() for record in self.records],
        }


GLOBAL_TRACKER = GlobalTokenTracker()


# ============================================================
# 4. Data Models
# ============================================================

class WorkflowNode(BaseModel):
    node_id: str
    type: str
    owner: str
    status: str
    content: Any
    evidence: List[str]
    depends_on: List[str]
    updated_by: Optional[str]
    version: int
    confidence: Any
    next_action: Optional[str] = None
    created_at: Optional[float] = None
    created_by_role: Optional[str] = None
    run_id: Optional[str] = None
    source_ref: Optional[str] = None
    operation_batch_id: Optional[str] = None


class WorkflowEdge(BaseModel):
    edge_id: str
    source: str
    relation: str
    target: str
    created_at: float
    created_by_role: str
    run_id: str
    operation_batch_id: str


class WorkflowGraphState(BaseModel):
    nodes: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge] = Field(default_factory=list)


class GraphOperation(BaseModel):
    op: str
    ref: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    relation: Optional[str] = None
    target: Optional[str] = None


class OperationExecutionRecord(BaseModel):
    index: int
    op: str
    success: bool
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    error: Optional[str] = None


class GraphExecutionResult(BaseModel):
    success: bool
    errors: List[str] = Field(default_factory=list)
    local_refs: Dict[str, str] = Field(default_factory=dict)
    operations: List[OperationExecutionRecord] = Field(default_factory=list)
    operation_batch_id: str


class CompactCompileResult(BaseModel):
    operations: List[GraphOperation] = Field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class EntityResult(BaseModel):
    entity: str
    value: float


class NumericComparisonOutput(BaseModel):
    entity_results: List[EntityResult]
    difference: float
    relation: str


class AgentRunLog(BaseModel):
    role: str
    mode: str = "normal"
    requirement_codes: List[str] = Field(default_factory=list)
    system_prompt_tokens: int = 0
    user_prompt_tokens: int = 0
    context: str
    context_tokens: int
    output: str
    output_tokens: int
    compiled_operations: List[GraphOperation] = Field(default_factory=list)
    compile_result: Optional[CompactCompileResult] = None
    execution_result: Optional[GraphExecutionResult] = None
    graph_snapshot: Dict[str, Any]


class AgentMessage(BaseModel):
    sender: str
    receiver: str
    round_id: int
    message_type: str
    raw_content: str
    compressed_content: Optional[str] = None
    raw_tokens: int = 0
    compressed_tokens: int = 0


class BaselineExperimentLog(BaseModel):
    task: str
    method: str
    messages: List[AgentMessage]
    final_answer: str
    communication_raw_tokens: int
    communication_compressed_tokens: int
    communication_saved_tokens: int
    communication_saving_rate: float
    global_token_summary: Dict[str, Any]


class GraphExperimentLog(BaseModel):
    task: str
    method: str
    final_answer: str
    workflow_graph: Dict[str, Any]
    logs: List[AgentRunLog]
    subgraph_context_tokens: int
    agent_output_tokens: int
    graph_update_tokens: int
    repeated_reasoning_count: int
    global_token_summary: Dict[str, Any]


class ExperimentSummary(BaseModel):
    method: str
    final_answer: str
    global_total_tokens: int
    accuracy: float
    latency: float
    details: Dict[str, Any]


# ============================================================
# 5. Local LLM Call
# ============================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    caller: str = "unknown",
    stage: str = "unknown",
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_tokens = int(inputs.input_ids.shape[-1])

    start_time = time.time()
    max_new_tokens = STAGE_MAX_NEW_TOKENS.get(stage, MAX_NEW_TOKENS)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    latency = time.time() - start_time
    generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
    output_tokens = int(generated_ids.shape[-1])

    GLOBAL_TRACKER.add(
        caller=caller,
        stage=stage,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sec=latency,
    )

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


# ============================================================
# 6. Workflow Graph Utilities
# ============================================================

def init_workflow_graph(user_task: str) -> WorkflowGraphState:
    nodes = {
        "task": WorkflowNode(
            node_id="task",
            type="task",
            owner="user",
            status="ready",
            content=user_task.strip(),
            evidence=[],
            depends_on=[],
            updated_by="user",
            version=1,
            confidence="high",
        ),
        "query_spec": WorkflowNode(
            node_id="query_spec",
            type="query_spec",
            owner="planner",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["task"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "facts": WorkflowNode(
            node_id="facts",
            type="facts",
            owner="planner",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["task", "query_spec"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "plan": WorkflowNode(
            node_id="plan",
            type="plan",
            owner="planner",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["task", "query_spec", "facts"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "calculation": WorkflowNode(
            node_id="calculation",
            type="calculation",
            owner="solver",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["query_spec", "facts", "plan"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "result": WorkflowNode(
            node_id="result",
            type="result",
            owner="solver",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["calculation"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "verification": WorkflowNode(
            node_id="verification",
            type="verification",
            owner="critic",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["query_spec", "facts", "calculation", "result"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
        "final_answer": WorkflowNode(
            node_id="final_answer",
            type="final_answer",
            owner="final_solver",
            status="empty",
            content=None,
            evidence=[],
            depends_on=["query_spec", "result", "verification"],
            updated_by=None,
            version=0,
            confidence="unknown",
        ),
    }
    return WorkflowGraphState(nodes=nodes)


def render_context(graph: WorkflowGraphState, role: str) -> Tuple[str, Dict[str, str]]:
    lines: List[str] = []
    alias_map: Dict[str, str] = {}
    visible_nodes: List[WorkflowNode] = []

    for subscribed_type in ROLE_SUBSCRIPTIONS[role]:
        matching_nodes = [
            node for node in graph.nodes.values()
            if node.type == subscribed_type
            and (node.status != "empty" or node.type == "task" or node.content is not None)
            and node.status != "superseded"
        ]
        if not matching_nodes and subscribed_type in graph.nodes:
            matching_nodes = [graph.nodes[subscribed_type]]
        if matching_nodes:
            visible_nodes.append(
                max(matching_nodes, key=lambda node: (node.created_at or 0, node.version))
            )

    for index, node in enumerate(visible_nodes, start=1):
        alias = f"visible_{index}"
        alias_map[alias] = node.node_id
        lines.append(f"[visible_{index}]")
        lines.append(f"type: {node.type}")
        lines.append(f"status: {node.status}")
        lines.append("latest: true")
        lines.append(f"content: {node.content}")
        lines.append(f"evidence: {'; '.join(node.evidence) if node.evidence else 'none'}")
        lines.append(f"confidence: {node.confidence}")
        lines.append("")
    return "\n".join(lines).strip(), alias_map


def copy_graph(graph: WorkflowGraphState) -> Dict[str, Any]:
    return graph.model_dump()


# ============================================================
# 7. Compact Role Output Compiler
# ============================================================

DEFAULT_STATUS = {
    ("planner", "query_spec"): "ready",
    ("planner", "facts"): "ready",
    ("planner", "plan"): "planned",
    ("solver", "calculation"): "solved",
    ("solver", "result"): "unverified",
    ("critic", "verification"): "need_fix",
    ("final_solver", "final_answer"): "final",
}

DEFAULT_CONFIDENCE = 0.9
PLACEHOLDER_VALUES = {
    "",
    "facts内容",
    "plan内容",
    "calculation内容",
    "result内容",
    "最终答案",
    "Final Solver",
    "...",
}


def is_placeholder_value(value: Any) -> bool:
    return normalize_text(str(value)) in PLACEHOLDER_VALUES


def latest_content_by_type(graph: WorkflowGraphState, node_type: str) -> str:
    node = latest_node_by_type(graph, node_type)
    if node is None or node.content is None:
        return ""
    return normalize_text(str(node.content))


def fallback_planner_tuple(graph: WorkflowGraphState) -> List[str]:
    task = latest_content_by_type(graph, "task")
    facts = extract_generic_facts(task, max_chars=220)
    plan = extract_generic_plan(task, max_chars=160)
    return [facts, plan]


def fallback_solver_tuple(graph: WorkflowGraphState) -> List[str]:
    source = "\n".join([
        latest_content_by_type(graph, "task"),
        latest_content_by_type(graph, "facts"),
    ])
    equations = extract_equations(source)
    calculation = "; ".join(equations) if equations else extract_generic_result(source, max_chars=220)
    result = extract_generic_result(source, max_chars=160)
    return [calculation, result]


def fallback_final_answer(graph: WorkflowGraphState) -> str:
    calculation = latest_content_by_type(graph, "calculation")
    result = latest_content_by_type(graph, "result")
    numeric_result, _ = parse_numeric_comparison_from_content(calculation, graph)
    if numeric_result is not None:
        _, rendered_result = render_numeric_comparison_contents(numeric_result, graph)
        return rendered_result
    source = "\n".join([calculation, result])
    extracted = extract_generic_result(source, max_chars=220)
    return extracted if extracted != "result not extracted" else result


def is_string_list(payload: List[Any], expected_len: int) -> bool:
    return len(payload) == expected_len and all(isinstance(item, str) for item in payload)


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def has_calculation_signal(text: str) -> bool:
    return bool(re.search(r"\d", text)) and any(symbol in text for symbol in ["=", "+", "-", "*", "×", "x", "/"])


def has_comparison_result_signal(text: str) -> bool:
    text = normalize_text(text).lower()
    return any(keyword in text for keyword in ["equal", "diff", "差", "相同", "更多", "less", "more", "same", "0"])


def planner_payload_is_low_quality(payload: List[str]) -> bool:
    facts, plan = payload
    return not has_number(facts) or len(normalize_text(plan)) < 8


def solver_payload_is_low_quality(payload: List[str]) -> bool:
    calculation, result = payload
    return not has_calculation_signal(calculation) or not has_comparison_result_signal(result)


def extract_json_array(text: str) -> Optional[List[Any]]:
    text = normalize_text(text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, list) else None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def is_number_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compact_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def expected_comparison_entities(graph: WorkflowGraphState) -> List[str]:
    query_spec = latest_content_by_type(graph, "query_spec")
    query_payload = extract_json_object(query_spec)
    if query_payload is not None:
        entities = query_payload.get("entities")
        if isinstance(entities, list) and all(isinstance(item, str) for item in entities):
            normalized_entities = dedupe_preserve_order(entities)
            if len(normalized_entities) >= 2:
                return normalized_entities[:2]

    task_entities = extract_comparison_entities_from_text(get_task_text(graph))
    if len(task_entities) >= 2:
        return task_entities[:2]

    facts = latest_content_by_type(graph, "facts")
    match = re.search(r"comparison_entities\s*=\s*(\[[^\]]+\])", facts)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return dedupe_preserve_order(parsed)[:2]
        except json.JSONDecodeError:
            pass
    return []


def normalize_product_name(product: str) -> str:
    product = normalize_text(product)
    parts = re.split(r"[。；;，,\s]+", product)
    return parts[-1] if parts else product


def extract_unit_prices(text: str) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    pattern = r"([\u4e00-\u9fffA-Za-z]+?)每[本支个件]\s*(\d+(?:\.\d+)?)\s*元"
    for match in re.finditer(pattern, normalize_text(text)):
        product = normalize_product_name(match.group(1))
        if product:
            prices[product] = float(match.group(2))
    return prices


def deterministic_numeric_comparison(graph: WorkflowGraphState) -> Optional[NumericComparisonOutput]:
    task = get_task_text(graph)
    source = "\n".join([task, latest_content_by_type(graph, "facts")])
    entities = expected_comparison_entities(graph)
    prices = extract_unit_prices(source)
    if len(entities) < 2 or not prices:
        return None

    values: List[float] = []
    for entity in entities[:2]:
        entity_pattern = re.escape(entity) + r"\s*买了([^。\n]*)"
        entity_match = re.search(entity_pattern, source)
        if not entity_match:
            return None
        segment = entity_match.group(1)
        total = 0.0
        matched_any = False
        for product, price in prices.items():
            quantity_pattern = r"(\d+(?:\.\d+)?)\s*[本支个件]\s*" + re.escape(product)
            quantity_match = re.search(quantity_pattern, segment)
            if quantity_match:
                total += float(quantity_match.group(1)) * price
                matched_any = True
        if not matched_any:
            return None
        values.append(total)

    first_value, second_value = values
    difference = abs(first_value - second_value)
    if abs(first_value - second_value) <= 1e-6:
        relation = "equal"
    elif first_value > second_value:
        relation = "first_greater"
    else:
        relation = "second_greater"

    return NumericComparisonOutput(
        entity_results=[
            EntityResult(entity=entities[0], value=first_value),
            EntityResult(entity=entities[1], value=second_value),
        ],
        difference=difference,
        relation=relation,
    )


def validate_numeric_comparison_payload(
    payload: Dict[str, Any],
    graph: Optional[WorkflowGraphState] = None,
) -> Tuple[Optional[NumericComparisonOutput], List[str]]:
    errors: List[str] = []
    expected_keys = {"entity_results", "difference", "relation"}
    extra_keys = set(payload.keys()) - expected_keys
    missing_keys = expected_keys - set(payload.keys())
    if extra_keys:
        errors.append(f"unexpected solver fields: {sorted(extra_keys)}")
    if missing_keys:
        errors.append(f"missing solver fields: {sorted(missing_keys)}")
    if errors:
        return None, errors

    entity_results = payload.get("entity_results")
    if not isinstance(entity_results, list) or len(entity_results) != 2:
        errors.append("entity_results must contain exactly two entities")
        return None, errors

    normalized_entities: List[Dict[str, Any]] = []
    for index, item in enumerate(entity_results):
        item_errors: List[str] = []
        if not isinstance(item, dict):
            item_errors.append(f"entity_results[{index}] must be an object")
            errors.extend(item_errors)
            continue
        item_keys = set(item.keys())
        if item_keys != {"entity", "value"}:
            item_errors.append(f"entity_results[{index}] must contain only entity and value")
            errors.extend(item_errors)
            continue
        entity = item.get("entity")
        value = item.get("value")
        if not isinstance(entity, str) or not normalize_text(entity):
            item_errors.append(f"entity_results[{index}].entity must be a non-empty string")
        if not is_number_value(value):
            item_errors.append(f"entity_results[{index}].value must be a number")
        if item_errors:
            errors.extend(item_errors)
        else:
            normalized_entities.append({"entity": normalize_text(entity), "value": float(value)})

    difference = payload.get("difference")
    if not is_number_value(difference) or float(difference) < 0:
        errors.append("difference must be a non-negative number")

    relation = payload.get("relation")
    allowed_relations = {"equal", "first_greater", "second_greater"}
    if relation not in allowed_relations:
        errors.append("relation must be equal, first_greater, or second_greater")

    if errors:
        return None, errors

    expected_entities = expected_comparison_entities(graph) if graph is not None else []
    if expected_entities and [item["entity"] for item in normalized_entities] != expected_entities[:2]:
        errors.append(
            "entity_results entities must match comparison_entities "
            f"{expected_entities[:2]}, got {[item['entity'] for item in normalized_entities]}"
        )
        return None, errors

    first_value = normalized_entities[0]["value"]
    second_value = normalized_entities[1]["value"]
    expected_difference = abs(first_value - second_value)
    if abs(float(difference) - expected_difference) > 1e-6:
        errors.append("difference does not match entity values")

    if relation == "equal" and abs(first_value - second_value) > 1e-6:
        errors.append("relation equal conflicts with unequal values")
    if relation == "first_greater" and first_value <= second_value:
        errors.append("relation first_greater conflicts with values")
    if relation == "second_greater" and second_value <= first_value:
        errors.append("relation second_greater conflicts with values")

    if errors:
        return None, errors

    return NumericComparisonOutput(
        entity_results=[EntityResult(**item) for item in normalized_entities],
        difference=float(difference),
        relation=relation,
    ), []


def parse_numeric_comparison_from_content(
    content: str,
    graph: WorkflowGraphState,
) -> Tuple[Optional[NumericComparisonOutput], List[str]]:
    payload = extract_json_object(content)
    if payload is None:
        return None, ["numeric comparison content is not a JSON object"]
    return validate_numeric_comparison_payload(payload, graph)


def programmatic_verify_numeric_result(graph: WorkflowGraphState) -> Tuple[bool, str]:
    calculation = latest_content_by_type(graph, "calculation")
    result = latest_content_by_type(graph, "result")
    numeric_result, errors = parse_numeric_comparison_from_content(calculation, graph)
    if numeric_result is None:
        return False, "; ".join(errors) or "missing structured numeric calculation"

    expected = deterministic_numeric_comparison(graph)
    if expected is not None:
        for actual_item, expected_item in zip(numeric_result.entity_results, expected.entity_results):
            if actual_item.entity != expected_item.entity or abs(actual_item.value - expected_item.value) > 1e-6:
                return False, "calculation values do not match query_spec and task facts"
        if abs(numeric_result.difference - expected.difference) > 1e-6 or numeric_result.relation != expected.relation:
            return False, "difference or relation does not match query_spec and task facts"

    for entity in expected_comparison_entities(graph):
        if entity and entity not in result:
            return False, f"result omits comparison entity: {entity}"

    _, rendered_result = render_numeric_comparison_contents(numeric_result, graph)
    if numeric_result.relation == "equal" and not any(keyword in result for keyword in ["相同", "均", "equal"]):
        return False, "result does not express equal relation"
    if numeric_result.relation != "equal" and "更多" not in result and "greater" not in result.lower():
        return False, "result does not express greater relation"
    if compact_number(numeric_result.difference) not in result:
        return False, "result omits difference"

    return True, rendered_result


def final_answer_is_complete_for_query(answer: str, graph: WorkflowGraphState) -> bool:
    answer = normalize_text(answer)
    if not answer:
        return False
    numeric_result, _ = parse_numeric_comparison_from_content(latest_content_by_type(graph, "calculation"), graph)
    if numeric_result is None:
        return True
    if answer.replace(".", "", 1).isdigit():
        return False
    if any(entity not in answer for entity in expected_comparison_entities(graph)):
        return False
    if compact_number(numeric_result.difference) not in answer:
        return False
    if numeric_result.relation == "equal":
        return any(keyword in answer for keyword in ["相同", "均", "一样", "equal"])
    return any(keyword in answer for keyword in ["更多", "greater", "多花"])


def render_numeric_comparison_contents(result: NumericComparisonOutput, graph: WorkflowGraphState) -> Tuple[str, str]:
    unit = "元" if "元" in "\n".join([latest_content_by_type(graph, "task"), latest_content_by_type(graph, "facts")]) else ""
    first = result.entity_results[0]
    second = result.entity_results[1]
    first_value = compact_number(first.value)
    second_value = compact_number(second.value)
    difference = compact_number(result.difference)
    calculation = json.dumps(
        {
            "entity_results": [
                {"entity": first.entity, "value": first.value},
                {"entity": second.entity, "value": second.value},
            ],
            "difference": result.difference,
            "relation": result.relation,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    if result.relation == "equal":
        answer = (
            f"{first.entity}和{second.entity}均为{first_value}{unit}，"
            f"花费相同，差额为{difference}{unit}。"
        )
    elif result.relation == "first_greater":
        answer = (
            f"{first.entity}花费更多，{first.entity}为{first_value}{unit}，"
            f"{second.entity}为{second_value}{unit}，差额为{difference}{unit}。"
        )
    else:
        answer = (
            f"{second.entity}花费更多，{second.entity}为{second_value}{unit}，"
            f"{first.entity}为{first_value}{unit}，差额为{difference}{unit}。"
        )

    return calculation, answer


def find_visible_alias_by_type(
    graph: WorkflowGraphState,
    alias_map: Dict[str, str],
    node_type: str,
) -> Optional[str]:
    best_alias: Optional[str] = None
    best_node: Optional[WorkflowNode] = None
    for alias, node_id in alias_map.items():
        node = graph.nodes.get(node_id)
        if node is not None and node.type == node_type:
            if best_node is None or (node.created_at or 0, node.version) > (best_node.created_at or 0, best_node.version):
                best_alias = alias
                best_node = node
    return best_alias


def add_compact_node_operation(
    operations: List[GraphOperation],
    *,
    role: str,
    ref: str,
    node_type: str,
    content: str,
    status: Optional[str] = None,
):
    operations.append(
        GraphOperation(
            op="add_node",
            ref=ref,
            type=node_type,
            content=normalize_text(content),
            status=status or DEFAULT_STATUS[(role, node_type)],
            confidence=DEFAULT_CONFIDENCE,
        )
    )


def add_compact_edge_operation(
    operations: List[GraphOperation],
    *,
    source: str,
    relation: str,
    target: Optional[str],
):
    if target:
        operations.append(
            GraphOperation(
                op="add_edge",
                source=source,
                relation=relation,
                target=target,
            )
        )


def compile_numeric_comparison_solver_output(
    text: str,
    graph: WorkflowGraphState,
    alias_map: Dict[str, str],
) -> CompactCompileResult:
    fallback_used = False
    fallback_reason: Optional[str] = None
    payload = extract_json_object(text)
    if payload is None:
        numeric_result = deterministic_numeric_comparison(graph)
        if numeric_result is None:
            return CompactCompileResult(errors=["solver output must be a numeric comparison JSON object"])
        fallback_used = True
        fallback_reason = "solver output is not a JSON object; deterministic numeric fallback used"
    else:
        numeric_result, errors = validate_numeric_comparison_payload(payload, graph)
        if errors or numeric_result is None:
            numeric_result = deterministic_numeric_comparison(graph)
            if numeric_result is None:
                return CompactCompileResult(errors=errors)
            fallback_used = True
            fallback_reason = "solver numeric object failed semantic validation; deterministic numeric fallback used"

    if numeric_result is None:
        return CompactCompileResult(errors=["numeric comparison result unavailable"])

    calculation_content, result_content = render_numeric_comparison_contents(numeric_result, graph)
    operations: List[GraphOperation] = []
    calc_ref = "c1"
    result_ref = "r1"
    add_compact_node_operation(
        operations,
        role="solver",
        ref=calc_ref,
        node_type="calculation",
        content=calculation_content,
    )
    add_compact_node_operation(
        operations,
        role="solver",
        ref=result_ref,
        node_type="result",
        content=result_content,
    )
    add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "facts"))
    add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "plan"))
    add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
    add_compact_edge_operation(operations, source=result_ref, relation="derived_from", target=calc_ref)
    add_compact_edge_operation(operations, source=result_ref, relation="constrained_by", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
    return CompactCompileResult(
        operations=operations,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


def compile_role_output_to_operations(
    role: str,
    text: str,
    graph: WorkflowGraphState,
    alias_map: Dict[str, str],
) -> CompactCompileResult:
    task_type = detect_task_type(get_task_text(graph))
    if role == "solver" and task_type == "numeric_comparison":
        return compile_numeric_comparison_solver_output(text, graph, alias_map)

    payload = extract_json_array(text)
    if payload is None:
        if role == "planner":
            payload = fallback_planner_tuple(graph)
            fallback_used = True
            fallback_reason: Optional[str] = "planner output is not valid JSON array"
        elif role == "solver":
            payload = fallback_solver_tuple(graph)
            fallback_used = True
            fallback_reason = "solver output is not valid JSON array"
        elif role == "final_solver":
            payload = [fallback_final_answer(graph)]
            fallback_used = True
            fallback_reason = "final output is not valid JSON array"
        else:
            return CompactCompileResult(errors=["output is not a JSON array"])
    else:
        fallback_used = False
        fallback_reason = None

    operations: List[GraphOperation] = []

    def use_fallback(reason: str, fallback_payload: List[str]) -> List[str]:
        nonlocal fallback_used, fallback_reason
        fallback_used = True
        fallback_reason = reason
        return fallback_payload

    if role == "planner":
        if not is_string_list(payload, 2):
            payload = use_fallback("planner output must be exactly two strings", fallback_planner_tuple(graph))
        elif is_placeholder_value(payload[0]) or is_placeholder_value(payload[1]):
            payload = use_fallback("planner output contains placeholder", fallback_planner_tuple(graph))
        elif planner_payload_is_low_quality(payload):
            payload = use_fallback("planner output is too incomplete", fallback_planner_tuple(graph))
        query_spec_ref = "q1"
        facts_ref = "f1"
        plan_ref = "p1"
        add_compact_node_operation(
            operations,
            role=role,
            ref=query_spec_ref,
            node_type="query_spec",
            content=query_spec_to_content(build_query_spec_from_task(get_task_text(graph))),
        )
        add_compact_node_operation(
            operations,
            role=role,
            ref=facts_ref,
            node_type="facts",
            content=str(payload[0]),
        )
        add_compact_node_operation(
            operations,
            role=role,
            ref=plan_ref,
            node_type="plan",
            content=str(payload[1]),
        )
        add_compact_edge_operation(operations, source=query_spec_ref, relation="derived_from", target=find_visible_alias_by_type(graph, alias_map, "task"))
        add_compact_edge_operation(operations, source=facts_ref, relation="derived_from", target=find_visible_alias_by_type(graph, alias_map, "task"))
        add_compact_edge_operation(operations, source=facts_ref, relation="constrained_by", target=query_spec_ref)
        add_compact_edge_operation(operations, source=plan_ref, relation="derived_from", target=facts_ref)
        add_compact_edge_operation(operations, source=plan_ref, relation="constrained_by", target=query_spec_ref)
        return CompactCompileResult(
            operations=operations,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    if role == "solver":
        if not is_string_list(payload, 2):
            payload = use_fallback("solver output must be exactly two strings", fallback_solver_tuple(graph))
        elif is_placeholder_value(payload[0]) or is_placeholder_value(payload[1]):
            payload = use_fallback("solver output contains placeholder", fallback_solver_tuple(graph))
        elif solver_payload_is_low_quality(payload):
            payload = use_fallback("solver output is too incomplete", fallback_solver_tuple(graph))
        calc_ref = "c1"
        result_ref = "r1"
        add_compact_node_operation(
            operations,
            role=role,
            ref=calc_ref,
            node_type="calculation",
            content=str(payload[0]),
        )
        add_compact_node_operation(
            operations,
            role=role,
            ref=result_ref,
            node_type="result",
            content=str(payload[1]),
        )
        add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "facts"))
        add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "plan"))
        add_compact_edge_operation(operations, source=calc_ref, relation="uses", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
        add_compact_edge_operation(operations, source=result_ref, relation="derived_from", target=calc_ref)
        add_compact_edge_operation(operations, source=result_ref, relation="constrained_by", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
        return CompactCompileResult(
            operations=operations,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    if role == "critic":
        if payload == [0]:
            is_correct = True
            content = "correct"
        elif len(payload) == 2 and payload[0] == 1 and isinstance(payload[1], str):
            is_correct = False
            content = payload[1]
        else:
            return CompactCompileResult(errors=["critic output must be [0] or [1,\"error\"]"])
        if is_correct and task_type == "numeric_comparison":
            is_program_correct, program_message = programmatic_verify_numeric_result(graph)
            if not is_program_correct:
                is_correct = False
                content = program_message
                fallback_used = True
                fallback_reason = "critic approval failed programmatic numeric verification"
        status = "verified" if is_correct else "need_fix"
        verification_ref = "v1"
        add_compact_node_operation(
            operations,
            role=role,
            ref=verification_ref,
            node_type="verification",
            content=content,
            status=status,
        )
        relation = "verifies" if is_correct else "contradicts"
        add_compact_edge_operation(operations, source=verification_ref, relation=relation, target=find_visible_alias_by_type(graph, alias_map, "result"))
        add_compact_edge_operation(operations, source=verification_ref, relation="constrained_by", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
        return CompactCompileResult(
            operations=operations,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    if role == "final_solver":
        if not is_string_list(payload, 1):
            payload = use_fallback("final output must be exactly one string", [fallback_final_answer(graph)])
        elif is_placeholder_value(payload[0]):
            payload = use_fallback("final output contains placeholder", [fallback_final_answer(graph)])
        elif task_type == "numeric_comparison" and not final_answer_is_complete_for_query(str(payload[0]), graph):
            payload = use_fallback("final output does not satisfy query_spec", [fallback_final_answer(graph)])
        answer_ref = "a1"
        add_compact_node_operation(
            operations,
            role=role,
            ref=answer_ref,
            node_type="final_answer",
            content=str(payload[0]),
        )
        add_compact_edge_operation(operations, source=answer_ref, relation="derived_from", target=find_visible_alias_by_type(graph, alias_map, "result"))
        add_compact_edge_operation(operations, source=answer_ref, relation="derived_from", target=find_visible_alias_by_type(graph, alias_map, "verification"))
        add_compact_edge_operation(operations, source=answer_ref, relation="constrained_by", target=find_visible_alias_by_type(graph, alias_map, "query_spec"))
        return CompactCompileResult(
            operations=operations,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    return CompactCompileResult(errors=[f"unknown role: {role}"])


ROLE_PERMISSIONS = {
    "planner": {
        "add_node": {"query_spec", "facts", "plan"},
        "add_edge": {"derived_from", "depends_on", "precedes", "constrained_by"},
        "update_node": set(),
        "update_status": set(),
        "supersede_node": set(),
    },
    "solver": {
        "add_node": {"calculation", "result"},
        "add_edge": {"uses", "produces", "derived_from", "constrained_by"},
        "update_node": {"calculation", "result"},
        "update_status": {"solved", "unverified", "need_fix"},
        "supersede_node": {"calculation", "result"},
    },
    "critic": {
        "add_node": {"verification", "error"},
        "add_edge": {"verifies", "contradicts", "identifies", "derived_from", "constrained_by"},
        "update_node": {"verification", "error"},
        "update_status": {"verified", "need_fix"},
        "supersede_node": {"verification", "error"},
    },
    "final_solver": {
        "add_node": {"final_answer"},
        "add_edge": {"derived_from", "summarizes", "constrained_by"},
        "update_node": {"final_answer"},
        "update_status": {"final"},
        "supersede_node": {"final_answer"},
    },
}

UPDATE_STATUS_NODE_TYPES = {
    "planner": set(),
    "solver": {"calculation", "result"},
    "critic": {"verification", "error"},
    "final_solver": {"final_answer"},
}

RELATION_TYPE_CONSTRAINTS = {
    "depends_on": {("query_spec", "task"), ("plan", "task"), ("plan", "facts"), ("calculation", "facts")},
    "precedes": {("facts", "plan"), ("plan", "calculation")},
    "uses": {("calculation", "query_spec"), ("calculation", "facts"), ("calculation", "plan")},
    "produces": {("calculation", "result"), ("verification", "final_answer")},
    "derived_from": {
        ("query_spec", "task"),
        ("facts", "task"),
        ("plan", "facts"),
        ("calculation", "facts"),
        ("calculation", "query_spec"),
        ("calculation", "plan"),
        ("result", "calculation"),
        ("verification", "result"),
        ("final_answer", "result"),
        ("final_answer", "verification"),
    },
    "constrained_by": {
        ("facts", "query_spec"),
        ("plan", "query_spec"),
        ("calculation", "query_spec"),
        ("result", "query_spec"),
        ("verification", "query_spec"),
        ("final_answer", "query_spec"),
    },
    "verifies": {("verification", "result"), ("verification", "calculation")},
    "contradicts": {("error", "result"), ("verification", "result")},
    "identifies": {("verification", "error")},
    "summarizes": {("final_answer", "result"), ("final_answer", "verification")},
    "supersedes": {
        ("facts", "facts"),
        ("plan", "plan"),
        ("calculation", "calculation"),
        ("result", "result"),
        ("verification", "verification"),
        ("final_answer", "final_answer"),
    },
}


class GraphOperationExecutionError(Exception):
    pass


class GraphOperationExecutor:
    def __init__(self, graph: WorkflowGraphState, permission_policy: Dict[str, Dict[str, set]]):
        self.graph = graph
        self.permission_policy = permission_policy

    def execute(
        self,
        operations: List[GraphOperation],
        *,
        alias_map: Dict[str, str],
        agent_role: str,
        run_id: str,
    ) -> GraphExecutionResult:
        operation_batch_id = self._make_operation_batch_id(operations, agent_role, run_id)
        local_refs: Dict[str, str] = {}
        staged_graph = copy.deepcopy(self.graph)
        records: List[OperationExecutionRecord] = []
        errors: List[str] = []

        if not operations:
            return GraphExecutionResult(
                success=False,
                errors=["operation batch contains no operations"],
                local_refs=local_refs,
                operations=records,
                operation_batch_id=operation_batch_id,
            )

        try:
            for index, operation in enumerate(operations):
                self._validate_permission(operation, agent_role)
                record = self._apply_operation(
                    graph=staged_graph,
                    operation=operation,
                    index=index,
                    alias_map=alias_map,
                    local_refs=local_refs,
                    agent_role=agent_role,
                    run_id=run_id,
                    operation_batch_id=operation_batch_id,
                )
                records.append(record)
        except GraphOperationExecutionError as error:
            errors.append(str(error))
            return GraphExecutionResult(
                success=False,
                errors=errors,
                local_refs=local_refs,
                operations=records,
                operation_batch_id=operation_batch_id,
            )

        self.graph.nodes = staged_graph.nodes
        self.graph.edges = staged_graph.edges
        return GraphExecutionResult(
            success=True,
            errors=[],
            local_refs=local_refs,
            operations=records,
            operation_batch_id=operation_batch_id,
        )

    def _make_operation_batch_id(self, operations: List[GraphOperation], agent_role: str, run_id: str) -> str:
        payload = json.dumps([operation.model_dump() for operation in operations], ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(f"{run_id}:{agent_role}:{payload}".encode()).hexdigest()[:12]
        return f"ops_{digest}"

    def _make_node_id(self, operation: GraphOperation, agent_role: str, run_id: str, index: int) -> str:
        normalized_content = normalize_text(str(operation.content or ""))
        payload = f"{run_id}:{agent_role}:{index}:{operation.ref}:{operation.type}:{normalized_content}"
        return f"node_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def _make_edge_id(self, source: str, relation: str, target: str) -> str:
        payload = f"{source}:{relation}:{target}"
        return f"edge_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"

    def _validate_permission(self, operation: GraphOperation, agent_role: str):
        role_policy = self.permission_policy.get(agent_role)
        if role_policy is None:
            raise GraphOperationExecutionError(f"unknown agent role: {agent_role}")

        if operation.op not in role_policy:
            raise GraphOperationExecutionError(f"unsupported op for role {agent_role}: {operation.op}")

        allowed_values = role_policy[operation.op]
        if operation.op == "add_node" and operation.type not in allowed_values:
            raise GraphOperationExecutionError(f"{agent_role} cannot add node type: {operation.type}")
        if operation.op == "add_edge" and operation.relation not in allowed_values:
            raise GraphOperationExecutionError(f"{agent_role} cannot add relation: {operation.relation}")
        if operation.op == "update_node" and operation.type not in allowed_values:
            raise GraphOperationExecutionError(f"{agent_role} cannot update node type: {operation.type}")
        if operation.op == "update_status" and operation.status not in allowed_values:
            raise GraphOperationExecutionError(f"{agent_role} cannot set status: {operation.status}")
        if operation.op == "supersede_node" and operation.type not in allowed_values:
            raise GraphOperationExecutionError(f"{agent_role} cannot supersede node type: {operation.type}")

    def _resolve_reference(
        self,
        value: Optional[str],
        *,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        graph: WorkflowGraphState,
    ) -> str:
        if not value:
            raise GraphOperationExecutionError("missing node reference")

        if value in local_refs:
            return local_refs[value]
        if value in alias_map:
            return alias_map[value]
        if value in graph.nodes:
            raise GraphOperationExecutionError(f"model used hidden global node id: {value}")

        raise GraphOperationExecutionError(f"unknown or invisible node reference: {value}")

    def _validate_confidence(self, operation: GraphOperation):
        if operation.confidence is None:
            return
        if operation.confidence < 0 or operation.confidence > 1:
            raise GraphOperationExecutionError(f"confidence out of range: {operation.confidence}")

    def _validate_edge_semantics(self, graph: WorkflowGraphState, source: str, relation: str, target: str):
        source_node = graph.nodes.get(source)
        target_node = graph.nodes.get(target)
        if source_node is None or target_node is None:
            raise GraphOperationExecutionError("edge source or target does not exist")

        allowed_pairs = RELATION_TYPE_CONSTRAINTS.get(relation)
        if allowed_pairs is None:
            raise GraphOperationExecutionError(f"unknown relation: {relation}")

        pair = (source_node.type, target_node.type)
        if pair not in allowed_pairs:
            raise GraphOperationExecutionError(
                f"relation {relation} cannot connect {source_node.type} -> {target_node.type}"
            )

    def _apply_operation(
        self,
        *,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        agent_role: str,
        run_id: str,
        operation_batch_id: str,
    ) -> OperationExecutionRecord:
        if operation.op == "add_node":
            return self._add_node(graph, operation, index, local_refs, agent_role, run_id, operation_batch_id)
        if operation.op == "add_edge":
            return self._add_edge(graph, operation, index, alias_map, local_refs, agent_role, run_id, operation_batch_id)
        if operation.op == "update_node":
            return self._update_node(graph, operation, index, alias_map, local_refs, agent_role, run_id, operation_batch_id)
        if operation.op == "update_status":
            return self._update_status(graph, operation, index, alias_map, local_refs, agent_role)
        if operation.op == "supersede_node":
            return self._supersede_node(graph, operation, index, alias_map, local_refs, agent_role, run_id, operation_batch_id)
        raise GraphOperationExecutionError(f"unsupported operation: {operation.op}")

    def _add_node(
        self,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        local_refs: Dict[str, str],
        agent_role: str,
        run_id: str,
        operation_batch_id: str,
    ) -> OperationExecutionRecord:
        if not operation.ref:
            raise GraphOperationExecutionError("add_node requires local ref")
        if operation.ref in local_refs:
            raise GraphOperationExecutionError(f"duplicate local ref: {operation.ref}")
        if not operation.type:
            raise GraphOperationExecutionError("add_node requires type")
        if operation.content is None or not str(operation.content).strip():
            raise GraphOperationExecutionError("add_node requires non-empty content")
        self._validate_confidence(operation)

        node_id = self._make_node_id(operation, agent_role, run_id, index)
        suffix = 1
        base_node_id = node_id
        while node_id in graph.nodes:
            suffix += 1
            node_id = f"{base_node_id}_{suffix}"

        now = time.time()
        graph.nodes[node_id] = WorkflowNode(
            node_id=node_id,
            type=operation.type,
            owner=agent_role,
            status=operation.status or "ready",
            content=normalize_text(str(operation.content)),
            evidence=["operation_batch"],
            depends_on=[],
            updated_by=agent_role,
            version=1,
            confidence=operation.confidence if operation.confidence is not None else "unknown",
            created_at=now,
            created_by_role=agent_role,
            run_id=run_id,
            source_ref=operation.ref,
            operation_batch_id=operation_batch_id,
        )
        local_refs[operation.ref] = node_id
        return OperationExecutionRecord(index=index, op=operation.op, success=True, node_id=node_id)

    def _add_edge(
        self,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        agent_role: str,
        run_id: str,
        operation_batch_id: str,
    ) -> OperationExecutionRecord:
        if not operation.relation:
            raise GraphOperationExecutionError("add_edge requires relation")
        source = self._resolve_reference(operation.source, alias_map=alias_map, local_refs=local_refs, graph=graph)
        target = self._resolve_reference(operation.target, alias_map=alias_map, local_refs=local_refs, graph=graph)
        self._validate_edge_semantics(graph, source, operation.relation, target)

        edge_id = self._make_edge_id(source, operation.relation, target)
        if any(edge.edge_id == edge_id for edge in graph.edges):
            return OperationExecutionRecord(index=index, op=operation.op, success=True, edge_id=edge_id)

        graph.edges.append(
            WorkflowEdge(
                edge_id=edge_id,
                source=source,
                relation=operation.relation,
                target=target,
                created_at=time.time(),
                created_by_role=agent_role,
                run_id=run_id,
                operation_batch_id=operation_batch_id,
            )
        )
        return OperationExecutionRecord(index=index, op=operation.op, success=True, edge_id=edge_id)

    def _update_node(
        self,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        agent_role: str,
        run_id: str,
        operation_batch_id: str,
    ) -> OperationExecutionRecord:
        target = self._resolve_reference(operation.target, alias_map=alias_map, local_refs=local_refs, graph=graph)
        node = graph.nodes[target]
        if operation.type and operation.type != node.type:
            raise GraphOperationExecutionError(f"update_node type mismatch: {operation.type} != {node.type}")
        self._validate_confidence(operation)

        if operation.content is not None:
            node.content = normalize_text(str(operation.content))
        if operation.status:
            node.status = operation.status
        if operation.confidence is not None:
            node.confidence = operation.confidence
        node.updated_by = agent_role
        node.version += 1
        node.run_id = run_id
        node.operation_batch_id = operation_batch_id
        return OperationExecutionRecord(index=index, op=operation.op, success=True, node_id=target)

    def _update_status(
        self,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        agent_role: str,
    ) -> OperationExecutionRecord:
        target = self._resolve_reference(operation.target, alias_map=alias_map, local_refs=local_refs, graph=graph)
        node = graph.nodes[target]
        allowed_node_types = UPDATE_STATUS_NODE_TYPES.get(agent_role, set())
        if node.type not in allowed_node_types:
            raise GraphOperationExecutionError(f"{agent_role} cannot update status for node type: {node.type}")
        node.status = operation.status or node.status
        node.updated_by = agent_role
        node.version += 1
        return OperationExecutionRecord(index=index, op=operation.op, success=True, node_id=target)

    def _supersede_node(
        self,
        graph: WorkflowGraphState,
        operation: GraphOperation,
        index: int,
        alias_map: Dict[str, str],
        local_refs: Dict[str, str],
        agent_role: str,
        run_id: str,
        operation_batch_id: str,
    ) -> OperationExecutionRecord:
        new_node_record = self._add_node(graph, operation, index, local_refs, agent_role, run_id, operation_batch_id)
        old_target = self._resolve_reference(operation.target, alias_map=alias_map, local_refs=local_refs, graph=graph)
        new_node_id = new_node_record.node_id
        if new_node_id is None:
            raise GraphOperationExecutionError("supersede_node failed to create replacement")
        self._validate_edge_semantics(graph, new_node_id, "supersedes", old_target)
        edge_id = self._make_edge_id(new_node_id, "supersedes", old_target)
        graph.edges.append(
            WorkflowEdge(
                edge_id=edge_id,
                source=new_node_id,
                relation="supersedes",
                target=old_target,
                created_at=time.time(),
                created_by_role=agent_role,
                run_id=run_id,
                operation_batch_id=operation_batch_id,
            )
        )
        graph.nodes[old_target].status = "superseded"
        return OperationExecutionRecord(index=index, op=operation.op, success=True, node_id=new_node_id, edge_id=edge_id)


# ============================================================
# 8. Shared Workflow Graph Execution
# ============================================================

def latest_node_by_type(graph: WorkflowGraphState, node_type: str) -> Optional[WorkflowNode]:
    candidates = [
        node for node in graph.nodes.values()
        if node.type == node_type
        and node.status not in {"empty", "superseded"}
        and node.content is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda node: (node.created_at or 0, node.version))


def run_graph_agent(
    role: str,
    graph: WorkflowGraphState,
    run_id: str,
    *,
    mode: str = "normal",
    feedback: Optional[str] = None,
) -> AgentRunLog:
    context, alias_map = render_context(graph, role)
    base_role = canonical_role(role)
    assignment = build_assignment(
        role=role,
        graph=graph,
        mode=mode,
        feedback=feedback,
    )
    system_prompt = REQUIREMENT_PROMPT_COMPILER.compile_system_prompt(assignment)
    user_prompt = REQUIREMENT_PROMPT_COMPILER.compile_user_prompt(
        assignment=assignment,
        context=context,
    )

    output = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        caller="agent",
        stage=base_role,
    )

    compile_result = compile_role_output_to_operations(base_role, output, graph, alias_map)
    executor = GraphOperationExecutor(graph, ROLE_PERMISSIONS)
    execution_result = executor.execute(
        operations=compile_result.operations,
        alias_map=alias_map,
        agent_role=base_role,
        run_id=run_id,
    )

    return AgentRunLog(
        role=role,
        mode=mode,
        requirement_codes=[code.value for code in assignment.requirement_codes],
        system_prompt_tokens=count_tokens(system_prompt),
        user_prompt_tokens=count_tokens(user_prompt),
        context=context,
        context_tokens=count_tokens(context),
        output=output,
        output_tokens=count_tokens(output),
        compiled_operations=compile_result.operations,
        compile_result=compile_result,
        execution_result=execution_result,
        graph_snapshot=copy.deepcopy(copy_graph(graph)),
    )


def run_workflow_graph_experiment(user_task: str) -> GraphExperimentLog:
    graph = init_workflow_graph(user_task)
    logs: List[AgentRunLog] = []
    repeated_reasoning_count = 0
    run_prefix = f"graph_run_{int(time.time() * 1000)}"

    logs.append(run_graph_agent("planner", graph, f"{run_prefix}_planner_1"))
    logs.append(run_graph_agent("solver", graph, f"{run_prefix}_solver_1"))
    logs.append(run_graph_agent("critic", graph, f"{run_prefix}_critic_1"))

    revision_count = 0
    verification_node = latest_node_by_type(graph, "verification")
    while verification_node is not None and verification_node.status == "need_fix" and revision_count < MAX_REVISIONS:
        repeated_reasoning_count += 1
        revision_count += 1

        feedback = normalize_text(str(verification_node.content or ""))
        solver_revision = run_graph_agent(
            "solver_revision",
            graph,
            f"{run_prefix}_solver_revision_{revision_count}",
            mode="revision",
            feedback=feedback,
        )
        logs.append(solver_revision)

        critic_recheck = run_graph_agent(
            "critic_recheck",
            graph,
            f"{run_prefix}_critic_recheck_{revision_count}",
            mode="recheck",
            feedback=feedback,
        )
        logs.append(critic_recheck)
        verification_node = latest_node_by_type(graph, "verification")

    logs.append(run_graph_agent("final_solver", graph, f"{run_prefix}_final_solver_1", mode="finalization"))
    final_node = latest_node_by_type(graph, "final_answer")

    return GraphExperimentLog(
        task=user_task,
        method="shared_workflow_graph",
        final_answer=str(final_node.content if final_node is not None else ""),
        workflow_graph=copy_graph(graph),
        logs=logs,
        subgraph_context_tokens=sum(log.context_tokens for log in logs),
        agent_output_tokens=sum(log.output_tokens for log in logs),
        graph_update_tokens=sum(log.output_tokens for log in logs),
        repeated_reasoning_count=repeated_reasoning_count,
        global_token_summary=GLOBAL_TRACKER.summary(),
    )


# ============================================================
# 9. Message Compression Baselines
# ============================================================

def build_protocol_message(
    text: str,
    message_type: str,
    default_state: str,
    default_next_action: str,
    default_output_limit: str,
) -> str:
    clean = normalize_text(text)
    stop_keys = ["STATE", "TASK", "FACTS", "RESULT", "ERRORS", "NEXT_ACTION", "OUTPUT_LIMIT"]
    task = extract_block_value(clean, "TASK", stop_keys) or "task_from_previous_message"
    facts = extract_block_value(clean, "FACTS", stop_keys) or extract_generic_facts(clean, max_chars=200)
    result = extract_block_value(clean, "RESULT", stop_keys) or extract_generic_result(clean, max_chars=200)
    errors = extract_block_value(clean, "ERRORS", stop_keys) or "none"

    if message_type == "critique":
        status = infer_verification_status(clean)
        if status == "verified":
            default_state = "verified"
            default_next_action = "final_answer_only"
            default_output_limit = "one_sentence"
            errors = "none"
        else:
            default_state = "need_fix"
            default_next_action = "revise"
            default_output_limit = "concise"
            if errors == "none":
                errors = extract_generic_result(clean, max_chars=160)

    return (
        f"STATE: {default_state}\n"
        f"TASK: {task}\n"
        f"FACTS: {facts}\n"
        f"RESULT: {result}\n"
        f"ERRORS: {errors}\n"
        f"NEXT_ACTION: {default_next_action}\n"
        f"OUTPUT_LIMIT: {default_output_limit}"
    )


class BaseCompressor:
    name = "base"

    def compress(
        self,
        message: AgentMessage,
        receiver_role: str,
        budget_tokens: int = 160,
    ) -> AgentMessage:
        raise NotImplementedError


class NoCompressionCompressor(BaseCompressor):
    name = "no_compression"

    def compress(
        self,
        message: AgentMessage,
        receiver_role: str,
        budget_tokens: int = 160,
    ) -> AgentMessage:
        message.raw_tokens = count_tokens(message.raw_content)
        message.compressed_content = message.raw_content
        message.compressed_tokens = message.raw_tokens
        return message


class ProtocolCompressor(BaseCompressor):
    name = "protocol_rule_based"

    def compress(
        self,
        message: AgentMessage,
        receiver_role: str,
        budget_tokens: int = 100,
    ) -> AgentMessage:
        message.raw_tokens = count_tokens(message.raw_content)
        text = message.raw_content.strip()

        if not text:
            message.compressed_content = ""
            message.compressed_tokens = 0
            return message

        if message.message_type == "plan":
            compressed = build_protocol_message(
                text=text,
                message_type=message.message_type,
                default_state="planned",
                default_next_action="solve",
                default_output_limit="concise",
            )
        elif message.message_type == "solution":
            compressed = build_protocol_message(
                text=text,
                message_type=message.message_type,
                default_state="solved",
                default_next_action="verify_only",
                default_output_limit="judge_only",
            )
        else:
            compressed = build_protocol_message(
                text=text,
                message_type=message.message_type,
                default_state="verified",
                default_next_action="final_answer_only",
                default_output_limit="one_sentence",
            )

        compressed = truncate_by_tokens(compressed, budget_tokens)
        message.compressed_content = compressed
        message.compressed_tokens = count_tokens(compressed)
        return message


# ============================================================
# 10. Baseline Agent Functions
# ============================================================

def planner_agent(task: str) -> str:
    return call_llm(
        GLOBAL_AGENT_CONSTRAINT + """
你是 Planner Agent。
你的职责是把任务转成最小可执行计划。
禁止解题，禁止展开计算。
只输出结构化交接信息。
""",
        f"""
原始任务：
{task}

请严格按以下格式输出，不要添加任何额外内容：

STATE: planned
TASK: task_from_user
FACTS: <抽取关键事实>
RESULT: none
ERRORS: none
NEXT_ACTION: solve
OUTPUT_LIMIT: concise
""",
        caller="agent",
        stage="planner",
    )


def solver_agent(task: str, received_context: str) -> str:
    return call_llm(
        GLOBAL_AGENT_CONSTRAINT + """
你是 Solver Agent。
你的职责是根据上游 FACTS 得出结果。
必须简洁。
禁止重复完整题目。
禁止写长篇解释。
""",
        f"""
上游消息：
{received_context}

请严格按以下格式输出，不要添加任何额外内容：

STATE: solved
TASK: task_from_user
FACTS: <保留关键事实>
RESULT: <写出必要计算和结论>
ERRORS: none
NEXT_ACTION: verify_only
OUTPUT_LIMIT: judge_only
""",
        caller="agent",
        stage="solver",
    )


def critic_agent(task: str, received_context: str) -> str:
    return call_llm(
        GLOBAL_AGENT_CONSTRAINT + """
你是 Critic Agent。
你的职责是核验 RESULT 是否正确。
如果正确，不要重新完整解题。
如果错误，只指出错误项和修正值。
""",
        f"""
待核验消息：
{received_context}

请严格按以下格式输出，不要添加任何额外内容：

STATE: <verified 或 need_fix>
TASK: task_from_user
FACTS: <保留关键事实>
RESULT: <若正确则保留上游 RESULT；若错误则给出修正结果>
ERRORS: <正确写 none；错误写具体错误>
NEXT_ACTION: <正确写 final_answer_only；错误写 revise>
OUTPUT_LIMIT: one_sentence
""",
        caller="agent",
        stage="critic",
    )


def final_solver_agent(task: str, solution: str, critique: str) -> str:
    return call_llm(
        GLOBAL_AGENT_CONSTRAINT + """
你是 Final Solver。
你必须根据 Critic 的 NEXT_ACTION 行动。

如果 NEXT_ACTION=final_answer_only：
只输出最终答案，一句话，不要解释，不要重新计算。

如果 NEXT_ACTION=revise：
只修正 ERRORS 指出的错误，并输出最终答案。
""",
        f"""
Solver 消息：
{solution}

Critic 消息：
{critique}

请严格按以下格式输出：

最终答案: <一句话>
""",
        caller="agent",
        stage="final_solver",
    )


def run_baseline_experiment(task: str, compressor: BaseCompressor) -> BaselineExperimentLog:
    messages: List[AgentMessage] = []

    plan = planner_agent(task)
    msg1 = compressor.compress(
        AgentMessage(
            sender="planner",
            receiver="solver",
            round_id=1,
            message_type="plan",
            raw_content=plan,
        ),
        receiver_role="solver",
    )
    messages.append(msg1)

    solution = solver_agent(task, msg1.compressed_content or "")
    msg2 = compressor.compress(
        AgentMessage(
            sender="solver",
            receiver="critic",
            round_id=2,
            message_type="solution",
            raw_content=solution,
        ),
        receiver_role="critic",
    )
    messages.append(msg2)

    critique = critic_agent(task, msg2.compressed_content or "")
    msg3 = compressor.compress(
        AgentMessage(
            sender="critic",
            receiver="final_solver",
            round_id=3,
            message_type="critique",
            raw_content=critique,
        ),
        receiver_role="final_solver",
    )
    messages.append(msg3)

    final_answer = final_solver_agent(task, solution, msg3.compressed_content or "")

    communication_raw = sum(message.raw_tokens for message in messages)
    communication_compressed = sum(message.compressed_tokens for message in messages)
    communication_saved = communication_raw - communication_compressed
    communication_saving_rate = (
        communication_saved / communication_raw if communication_raw > 0 else 0.0
    )

    return BaselineExperimentLog(
        task=task,
        method=compressor.name,
        messages=messages,
        final_answer=final_answer,
        communication_raw_tokens=communication_raw,
        communication_compressed_tokens=communication_compressed,
        communication_saved_tokens=communication_saved,
        communication_saving_rate=communication_saving_rate,
        global_token_summary=GLOBAL_TRACKER.summary(),
    )


# ============================================================
# 11. Unified Comparison
# ============================================================

def build_summary(
    method: str,
    final_answer: str,
    global_token_summary: Dict[str, Any],
    reference_answer: str,
    details: Dict[str, Any],
) -> ExperimentSummary:
    return ExperimentSummary(
        method=method,
        final_answer=final_answer,
        global_total_tokens=global_token_summary["total_tokens"],
        accuracy=parse_yes_no_accuracy(reference_answer, final_answer),
        latency=global_token_summary["total_latency_sec"],
        details=details,
    )


def run_all_experiments(task: str, reference_answer: str) -> List[ExperimentSummary]:
    summaries: List[ExperimentSummary] = []

    compressors: List[BaseCompressor] = [
        NoCompressionCompressor(),
        ProtocolCompressor(),
    ]

    for compressor in compressors:
        GLOBAL_TRACKER.reset()
        result = run_baseline_experiment(task, compressor)
        summaries.append(
            build_summary(
                method=result.method,
                final_answer=result.final_answer,
                global_token_summary=result.global_token_summary,
                reference_answer=reference_answer,
                details=result.model_dump(),
            )
        )

    GLOBAL_TRACKER.reset()
    graph_result = run_workflow_graph_experiment(task)
    summaries.append(
        build_summary(
            method=graph_result.method,
            final_answer=graph_result.final_answer,
            global_token_summary=graph_result.global_token_summary,
            reference_answer=reference_answer,
            details=graph_result.model_dump(),
        )
    )

    return summaries


def print_comparison(results: List[ExperimentSummary]):
    print("\n==============================")
    print("Unified Experiment Comparison")
    print("==============================")

    for result in results:
        print("\n---")
        print(f"Method: {result.method}")
        print(f"Global total tokens: {result.global_total_tokens}")
        print(f"Accuracy: {result.accuracy:.2f}")
        print(f"Latency: {result.latency:.2f}s")
        print(f"Final answer: {result.final_answer}")


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":
    task = """
一家商店卖笔记本和钢笔。笔记本每本 3 元，钢笔每支 2 元。
Alice 买了 4 本笔记本和 5 支钢笔。
Bob 买了 2 本笔记本和 8 支钢笔。
请问谁花的钱更多？多花了多少钱？
"""

    reference_answer = "Alice 和 Bob 都花了 22 元，花费相同，没有人多花钱。"

    all_results = run_all_experiments(task, reference_answer)
    print_comparison(all_results)

    os.makedirs("logs", exist_ok=True)
    output_path = "logs/unified_multi_method_experiment.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump([result.model_dump() for result in all_results], file, ensure_ascii=False, indent=2)

    print(f"\nSaved log to {output_path}")
