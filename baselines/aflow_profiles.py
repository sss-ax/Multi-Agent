"""Dataset/workflow-specific contracts for the AFlow graph adapter.

The graph runtime is intentionally generic, but AFlow's archived workflows are
not.  This module is the policy boundary: a profile decides which outputs are
candidates, which invocations may be replayed, and how a receiver's native
operator arguments are reconstructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ReceiverContract:
    receiver: str
    required_artifact_types: tuple[str, ...] = ()
    optional_artifact_types: tuple[str, ...] = ()
    forbidden_artifact_types: tuple[str, ...] = ()
    required_source_types: tuple[str, ...] = ()
    required_relations: tuple[str, ...] = ()
    preserve_multiplicity: bool = True
    include_source_refs: bool = False
    dependency_closure: bool = True
    max_tokens: int = 0
    renderer: str = "native"
    fallback: str = "original"
    required_bindings: tuple["RequiredBindingSpec", ...] = ()


@dataclass(frozen=True)
class RequiredBindingSpec:
    """Field-level requirements used by the v4.2 replay gate."""
    field_name: str
    requirement_class: str
    value_kind: str = "raw"
    cardinality: str = "one"
    ordered: bool = False
    preserve_multiplicity: bool = False


@dataclass(frozen=True)
class AFlowOperatorInput:
    kwargs: dict[str, Any]
    source_node_ids: tuple[str, ...] = ()
    context: Any = None
    artifact_binding_hints: Mapping[str, tuple[str, ...]] = None
    candidate_batch_id: str | None = None
    invocation_id: str | None = None
    source_binding_hints: Mapping[str, tuple[str, ...]] = None


@dataclass(frozen=True)
class AFlowDeliveredArtifact:
    node_id: str
    producer_operator: str
    artifact_kind: str
    candidate_role: str
    multiplicity: int
    value: Any


@dataclass(frozen=True)
class AFlowOptimizationProfile:
    dataset: str
    workflow_round: int
    candidate_operators: tuple[str, ...]
    ensemble_operators: tuple[str, ...]
    reusable_invocations: tuple[str, ...]
    output_normalizer: Callable[[Any], Any]
    input_binding: Callable[..., AFlowOperatorInput]
    candidate_signature: Callable[..., str]
    validated: bool = False
    name: str = ""
    version: str = "v1"
    invocation_predicate: Callable[..., bool] | None = None
    source_selector: Callable[..., tuple[str, ...]] | None = None
    candidate_projection: Callable[..., Any] | None = None
    ensemble_strategy: str = "native"
    ensemble_prompt_builder: Callable[..., tuple[str, dict[str, str]]] | None = None
    ensemble_fill_mode: str | None = None
    supports_content_dedup: bool = True
    supports_invocation_reuse: bool = True
    # Context compression is deliberately opt-in.  Profiles that do not
    # provide a lossless source binding continue to use their native inputs.
    supports_context_compression: bool = False
    receiver_contracts: Mapping[str, ReceiverContract] = None
    operator_roles: Mapping[str, str] = None

    @property
    def profile_name(self) -> str:
        return self.name or f"{self.dataset}_round{self.workflow_round}_{self.version}"

    def matches_operator(self, configured: tuple[str, ...], operator: str) -> bool:
        value = str(operator).casefold()
        return any(value == item.casefold() for item in configured)

    def is_candidate_operator(self, operator: str) -> bool:
        return self.matches_operator(self.candidate_operators, operator)

    def is_ensemble_operator(self, operator: str) -> bool:
        return self.matches_operator(self.ensemble_operators, operator)

    def allows_reuse(self, operator: str, kwargs: Mapping[str, Any] | None = None) -> bool:
        if not self.matches_operator(self.reusable_invocations, operator):
            return False
        if self.invocation_predicate is not None:
            return bool(self.invocation_predicate(operator=operator, kwargs=kwargs or {}))
        return True

    def receiver_contract(self, operator: str) -> ReceiverContract:
        contracts = self.receiver_contracts or {}
        for name, contract in contracts.items():
            if str(name).casefold() == str(operator).casefold():
                return contract
        return ReceiverContract(receiver=str(operator))

    def operator_role(self, operator: str, kwargs: Mapping[str, Any] | None = None) -> str:
        for name, role in (self.operator_roles or {}).items():
            if str(name).casefold() == str(operator).casefold():
                return str(role)
        return "passthrough"


def _identity_output(value: Any) -> Any:
    return value


def _operator_text(kwargs: Mapping[str, Any]) -> str:
    return str(kwargs.get("instruction", "")).casefold()


def _source_ids(artifacts: Any) -> tuple[str, ...]:
    if artifacts is None:
        return ()
    if isinstance(artifacts, Mapping):
        artifacts = artifacts.values()
    ids = []
    for item in artifacts:
        node_id = getattr(item, "node_id", None)
        if node_id is None and isinstance(item, Mapping):
            node_id = item.get("node_id", "")
        if node_id:
            ids.append(str(node_id))
    return tuple(ids)


def _gsm8k_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    # GSM8K's native graph already supplies the correct prompt.  The graph
    # layer only replaces candidate lists and selected analysis.
    kwargs = dict(original_kwargs)
    ids = _source_ids(delivered_artifacts)
    if session.profile_is_ensemble(operator):
        if "solutions" in kwargs:
            kwargs["solutions"] = [session.operator_value_for_node(node_id) for node_id in ids]
    elif str(operator).casefold() == "programmer" and ids and "analysis" in kwargs:
        kwargs["analysis"] = session.operator_value_for_node(ids[-1])
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _drop_source_selector(*, session: Any, operator: str, original_kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    lowered = str(operator).casefold()
    if lowered == "answergenerate":
        return ()
    if lowered == "scensemble":
        return session.candidate_pool_ids()
    if lowered == "custom":
        return session.latest_output_ids()
    return ()


def _drop_candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    return "answer"


def _drop_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    """Bind DROP's native arguments without rendering generic graph text."""
    kwargs = dict(original_kwargs)
    artifacts = tuple(delivered_artifacts or ())
    ids = tuple(str(item.node_id) for item in artifacts)
    lowered = str(operator).casefold()
    if lowered == "scensemble" and "solutions" in kwargs:
        # A single canonical artifact is safe to collapse.  If multiple
        # artifacts remain, expand their multiplicities so self-consistency
        # voting keeps the original weights.
        if len(artifacts) == 1:
            solutions = [artifacts[0].value]
        else:
            solutions = []
            for artifact in artifacts:
                solutions.extend([artifact.value] * max(1, int(artifact.multiplicity)))
        kwargs["solutions"] = solutions
    elif lowered == "custom" and artifacts and "input" in kwargs:
        native_input = str(kwargs["input"])
        marker = "\nBest solution:"
        if marker in native_input:
            problem = native_input.split(marker, 1)[0]
            if problem.startswith("Question: "):
                problem = problem[len("Question: "):]
            kwargs["input"] = f"Question: {problem}\nBest solution: {artifacts[-1].value}"
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _hotpotqa_source_selector(*, session: Any, operator: str, original_kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    lowered = str(operator).casefold()
    if lowered == "answergenerate":
        return ()
    if lowered == "scensemble":
        return session.candidate_pool_ids()
    if lowered == "custom":
        return session.latest_output_ids()
    return ()


def _hotpotqa_candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    return "answer"


def _hotpotqa_candidate_projection(*, output: Any, **_: Any) -> Any:
    """Hotpot's candidate is the answer, while thought stays provenance."""
    if isinstance(output, Mapping) and "answer" in output:
        return output["answer"]
    return output


def _hotpotqa_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    """Bind HotpotQA's native answer-only ensemble and final format prompt."""
    kwargs = dict(original_kwargs)
    artifacts = tuple(delivered_artifacts or ())
    ids = tuple(str(item.node_id) for item in artifacts)
    lowered = str(operator).casefold()
    if lowered == "scensemble" and "solutions" in kwargs:
        if len(artifacts) == 1:
            solutions = [artifacts[0].value]
        else:
            solutions = []
            for artifact in artifacts:
                solutions.extend([artifact.value] * max(1, int(artifact.multiplicity)))
        kwargs["solutions"] = solutions
    elif lowered == "custom" and artifacts and "input" in kwargs:
        native_input = str(kwargs["input"])
        marker = "\nBest answer:"
        if marker in native_input:
            problem = native_input.split(marker, 1)[0]
            if problem.startswith("Question: "):
                problem = problem[len("Question: "):]
            kwargs["input"] = f"Question: {problem}\nBest answer: {artifacts[-1].value}"
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _mbpp_source_selector(*, session: Any, operator: str, original_kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    lowered = str(operator).casefold()
    if lowered == "customcodegenerate":
        return ()
    if lowered == "scensemble":
        return session.candidate_pool_ids()
    if lowered in {"test", "custom"}:
        return session.latest_output_ids()
    return ()


def _mbpp_candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    return "code_solution"


def _mbpp_candidate_projection(*, output: Any, **_: Any) -> Any:
    if isinstance(output, Mapping) and "response" in output:
        return output["response"]
    return output


def _mbpp_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    """Preserve MBPP's native problem/test/fix contracts."""
    kwargs = dict(original_kwargs)
    artifacts = tuple(delivered_artifacts or ())
    ids = tuple(str(item.node_id) for item in artifacts)
    lowered = str(operator).casefold()
    candidate_artifacts = tuple(
        item for item in artifacts
        if str(getattr(item, "candidate_role", "")) == "code_solution"
    )
    if lowered == "scensemble" and "solutions" in kwargs:
        kwargs["solutions"] = [item.value for item in candidate_artifacts]
    elif lowered == "test" and artifacts and "solution" in kwargs:
        kwargs["solution"] = artifacts[-1].value
    # Custom receives a native Problem/Failed solution/Error string.  Do not
    # render a generic graph context over it.
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _humaneval_source_selector(*, session: Any, operator: str, original_kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    lowered = str(operator).casefold()
    if lowered == "customcodegenerate":
        # Initial generation and retry both preserve their native arguments.
        # The retry does not implicitly receive a graph-rendered prompt.
        return ()
    if lowered == "test":
        return session.latest_output_ids()
    return ()


def _humaneval_candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    instruction = str(kwargs.get("instruction", "")).strip()
    return "code_repair" if instruction else "code_initial"


def _humaneval_candidate_projection(*, output: Any, **_: Any) -> Any:
    if isinstance(output, Mapping) and "response" in output:
        return output["response"]
    return output


def _humaneval_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    """Preserve HumanEval's generation, local test, and retry contracts."""
    kwargs = dict(original_kwargs)
    raw_artifacts = tuple(delivered_artifacts or ())
    artifacts = tuple(item for item in raw_artifacts if hasattr(item, "node_id"))
    mapping_ids = tuple(
        str(item.get("node_id"))
        for item in raw_artifacts
        if isinstance(item, Mapping) and item.get("node_id")
    )
    if mapping_ids:
        artifacts = tuple((*artifacts, *session.delivered_artifacts(mapping_ids)))
    ids = tuple(str(item.node_id) for item in artifacts)
    if str(operator).casefold() == "test" and artifacts and "solution" in kwargs:
        kwargs["solution"] = artifacts[-1].value
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _mbpp_weighted_prompt(*, session: Any, candidates: tuple[tuple[str, int], ...], original_kwargs: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    blocks = []
    for index, (code, multiplicity) in enumerate(candidates):
        letter = chr(65 + index)
        mapping[letter] = code
        blocks.append(
            f"{letter} (x{multiplicity}):\n{code}"
        )
    problem = str(original_kwargs.get("problem") or session.task)
    entry_point = str(
        original_kwargs.get("entry_point")
        or session.task_metadata.get("entry_point", "")
    )
    if not entry_point:
        session.telemetry.emit("aflow_weighted_prompt_missing_entry_point", {
            "task_id": session.task_id,
            "dataset": session.dataset,
        })
    prompt = (
        f"Problem: {problem}\n"
        f"Entry point: {entry_point}\n"
        "Candidates:\n"
        + "\n\n".join(blocks)
        + f"\n\nChoose the highest count; ties use first. Reply with one ID only: {', '.join(mapping)}."
    )
    return prompt, mapping


def _math_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    kwargs = dict(original_kwargs)
    raw_artifacts = tuple(delivered_artifacts or ())
    artifacts = tuple(
        item
        for item in raw_artifacts
        if hasattr(item, "node_id") and hasattr(item, "producer_operator")
    )
    mapping_ids = tuple(
        str(item.get("node_id"))
        for item in raw_artifacts
        if isinstance(item, Mapping) and item.get("node_id")
    )
    if mapping_ids:
        artifacts = tuple((*artifacts, *session.delivered_artifacts(mapping_ids)))
    ids = tuple(str(item.node_id) for item in artifacts)
    lowered = str(operator).casefold()
    role = _math_custom_role(kwargs) if lowered == "custom" else ""
    if lowered == "custom" and role in {"detailed", "generated"}:
        return AFlowOperatorInput(kwargs=kwargs, source_node_ids=())
    if lowered == "custom" and role == "refined":
        programmer = next(
            (item for item in artifacts if item.producer_operator.casefold() == "programmer"),
            None,
        )
        if programmer is None:
            raise RuntimeError("MATH Refine Custom requires Programmer execution output")
        problem = session.original_problem(kwargs)
        kwargs["input"] = f"{problem}\nCode output: {programmer.value}"
        return AFlowOperatorInput(kwargs=kwargs, source_node_ids=(programmer.node_id,))
    if lowered == "scensemble" and "solutions" in kwargs:
        candidates = tuple(
            item for item in artifacts
            if item.candidate_role in {"refined", "detailed", "generated"}
        )
        kwargs["solutions"] = [item.value for item in candidates]
        return AFlowOperatorInput(kwargs=kwargs, source_node_ids=tuple(item.node_id for item in candidates))
    return AFlowOperatorInput(kwargs=kwargs, source_node_ids=ids)


def _math_custom_role(kwargs: Mapping[str, Any]) -> str:
    instruction = _normalize_math_instruction(str(kwargs.get("instruction", "")))
    symbolic_roles = {
        "refine_answer_prompt": "refined",
        "detailed_solution_prompt": "detailed",
        "generate_solution_prompt": "generated",
    }
    if instruction in symbolic_roles:
        return symbolic_roles[instruction]

    # Archived MATH workflows pass the expanded prompt text rather than the
    # symbolic constant name.  Match its stable, unique opening sentence;
    # never classify arbitrary occurrences of words such as "detailed" or
    # "generate" in the body of an unknown prompt.
    prompt_prefixes = (
        ("Given the mathematical problem and the output from the code execution,", "refined"),
        ("Provide a comprehensive, step-by-step solution to the given mathematical problem.", "detailed"),
        ("Please solve the given mathematical problem step by step.", "generated"),
    )
    for prefix, role in prompt_prefixes:
        if instruction.startswith(prefix.casefold()):
            return role
    raise ValueError(f"unknown MATH Custom instruction: {kwargs.get('instruction', '')!r}")


def _normalize_math_instruction(value: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in value.replace("\r\n", "\n").strip().splitlines()
    ).casefold()


def _math_source_selector(*, session: Any, operator: str, original_kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    lowered = str(operator).casefold()
    if lowered == "programmer":
        return ()
    if lowered == "scensemble":
        return session.candidate_pool_ids()
    if lowered == "custom":
        role = _math_custom_role(original_kwargs)
        return session.latest_output_ids() if role == "refined" else ()
    return ()


def _math_candidate_projection(*, output: Any, **_: Any) -> Any:
    if isinstance(output, Mapping) and "response" in output:
        return output["response"]
    return output


def _math_weighted_prompt(*, session: Any, candidates: tuple[tuple[str, int], ...], original_kwargs: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    blocks = []
    node_ids = session.candidate_pool_ids()
    for index, (value, multiplicity) in enumerate(candidates):
        letter = chr(65 + index)
        mapping[letter] = value
        role = "candidate"
        if index < len(node_ids):
            role = str(session.store.node(node_ids[index]).provenance.get("candidate_role", "candidate"))
        blocks.append(f"{letter} [{role}] (x{multiplicity}):\n{value}")
    problem = str(original_kwargs.get("problem") or session.task)
    prompt = (
        f"Problem: {problem}\n\n"
        + "\n\n".join(blocks)
        + f"\n\nChoose the most consistent answer using counts as vote weights. Reply with one ID only: {', '.join(mapping)}."
    )
    return prompt, mapping


def _generic_binding(*, session: Any, operator: str, original_kwargs: Mapping[str, Any], delivered_artifacts: Any) -> AFlowOperatorInput:
    # Defined for audit/experimental use, but never selected implicitly for an
    # unvalidated profile.  It deliberately does not rewrite arbitrary input.
    return AFlowOperatorInput(kwargs=dict(original_kwargs), source_node_ids=_source_ids(delivered_artifacts))


def _candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    return str(operator).casefold() + "|" + _operator_text(kwargs)


def _math_candidate_signature(*, operator: str, kwargs: Mapping[str, Any], **_: Any) -> str:
    return _math_custom_role(kwargs)


def _math_reuse_predicate(*, operator: str, kwargs: Mapping[str, Any]) -> bool:
    return str(operator).casefold() == "custom" and _math_custom_role(kwargs) == "generated"


def _profile(dataset: str, round_index: int, *, candidates: tuple[str, ...], ensembles: tuple[str, ...], reusable: tuple[str, ...], binding: Callable[..., AFlowOperatorInput], signature: Callable[..., str], validated: bool = False, invocation_predicate: Callable[..., bool] | None = None, source_selector: Callable[..., tuple[str, ...]] | None = None, candidate_projection: Callable[..., Any] | None = None, ensemble_strategy: str = "native", ensemble_prompt_builder: Callable[..., tuple[str, dict[str, str]]] | None = None, ensemble_fill_mode: str | None = None, supports_content_dedup: bool = True, supports_invocation_reuse: bool = True, supports_context_compression: bool = False, receiver_contracts: Mapping[str, ReceiverContract] | None = None, operator_roles: Mapping[str, str] | None = None) -> AFlowOptimizationProfile:
    return AFlowOptimizationProfile(
        dataset=dataset,
        workflow_round=round_index,
        candidate_operators=candidates,
        ensemble_operators=ensembles,
        reusable_invocations=reusable,
        output_normalizer=_identity_output,
        input_binding=binding,
        candidate_signature=signature,
        validated=validated,
        name=f"{dataset}_round{round_index}_v1",
        invocation_predicate=invocation_predicate,
        source_selector=source_selector,
        candidate_projection=candidate_projection,
        ensemble_strategy=ensemble_strategy,
        ensemble_prompt_builder=ensemble_prompt_builder,
        ensemble_fill_mode=ensemble_fill_mode,
        supports_content_dedup=supports_content_dedup,
        supports_invocation_reuse=supports_invocation_reuse,
        supports_context_compression=supports_context_compression,
        receiver_contracts=receiver_contracts or {},
        operator_roles=operator_roles or {},
    )


_HOTPOT_CONTRACTS = {
    "AnswerGenerate": ReceiverContract("AnswerGenerate", ("evidence_ref", "opaque_text"), ("constraint", "uncertainty"), required_source_types=("question_source", "sentence_source"), include_source_refs=True),
    "ScEnsemble": ReceiverContract("ScEnsemble", ("candidate_answer",), ("confidence", "uncertainty", "evidence_ref"), forbidden_artifact_types=("reasoning_summary", "opaque_text"), preserve_multiplicity=True),
    "Custom": ReceiverContract("Custom", ("candidate_answer",), ("evidence_ref", "uncertainty"), forbidden_artifact_types=("reasoning_summary", "opaque_text"), preserve_multiplicity=True),
}

_MBPP_CONTRACTS = {
    "ScEnsemble": ReceiverContract(
        "ScEnsemble", ("code",),
        forbidden_artifact_types=("reasoning_summary", "test_result", "opaque_text"),
        preserve_multiplicity=True, renderer="mbpp_ensemble",
        required_bindings=(RequiredBindingSpec("solutions", "required_artifact", "artifact", "exact_batch", True, True),),
    ),
    "Test": ReceiverContract(
        "Test", ("code",), required_source_types=("problem", "entry_point"), renderer="mbpp_test",
        required_bindings=(
            RequiredBindingSpec("solution", "required_artifact", "artifact", "one"),
            RequiredBindingSpec("problem", "required_source", "source", "one"),
            RequiredBindingSpec("entry_point", "required_source", "source", "one"),
            RequiredBindingSpec("tests", "required_source", "source", "zero_or_one"),
        ),
    ),
    "Custom": ReceiverContract(
        "Custom", ("code", "test_result"), optional_artifact_types=("critique",),
        required_source_types=("problem",), forbidden_artifact_types=("opaque_text",), renderer="mbpp_fix",
        required_bindings=(
            RequiredBindingSpec("failed_code", "required_artifact", "artifact", "one"),
            RequiredBindingSpec("test_result", "required_artifact", "artifact", "one"),
            RequiredBindingSpec("problem", "required_source", "source", "one"),
        ),
    ),
}


_PROFILES = {
    ("GSM8K", 10): _profile("GSM8K", 10, candidates=("Custom",), ensembles=("ScEnsemble",), reusable=("Custom",), binding=_gsm8k_binding, signature=_candidate_signature, validated=True),
    ("DROP", 3): _profile("DROP", 3, candidates=("AnswerGenerate",), ensembles=("ScEnsemble",), reusable=("AnswerGenerate",), binding=_drop_binding, signature=_drop_candidate_signature, source_selector=_drop_source_selector, validated=True),
    # Validated on the paired 10-row native/shadow/sendall/weighted runs:
    # weighted ensemble + exact candidate projection + deterministic reuse.
    ("HotpotQA", 3): _profile("HotpotQA", 3, candidates=("AnswerGenerate",), ensembles=("ScEnsemble",), reusable=("AnswerGenerate",), binding=_hotpotqa_binding, signature=_hotpotqa_candidate_signature, source_selector=_hotpotqa_source_selector, candidate_projection=_hotpotqa_candidate_projection, ensemble_strategy="weighted", validated=True, supports_context_compression=True, receiver_contracts=_HOTPOT_CONTRACTS, operator_roles={"AnswerGenerate": "source_producer", "ScEnsemble": "artifact_receiver", "Custom": "hybrid_receiver"}),
("MATH", 5): _profile("MATH", 5, candidates=("Custom",), ensembles=("ScEnsemble",), reusable=("Custom",), binding=_math_binding, signature=_math_candidate_signature, source_selector=_math_source_selector, candidate_projection=_math_candidate_projection, invocation_predicate=_math_reuse_predicate, ensemble_strategy="native_multiplicity", ensemble_prompt_builder=_math_weighted_prompt, ensemble_fill_mode="context_fill"),
    # Validated on the paired 50-row run with PYTHONHASHSEED=0: score parity,
    # deterministic reuse, exact code deduplication, and zero weighted fallbacks.
    ("MBPP", 14): _profile("MBPP", 14, candidates=("CustomCodeGenerate",), ensembles=("ScEnsemble",), reusable=("CustomCodeGenerate",), binding=_mbpp_binding, signature=_mbpp_candidate_signature, source_selector=_mbpp_source_selector, candidate_projection=_mbpp_candidate_projection, ensemble_strategy="weighted_code", ensemble_prompt_builder=_mbpp_weighted_prompt, ensemble_fill_mode="context_fill", validated=True, receiver_contracts=_MBPP_CONTRACTS, operator_roles={"CustomCodeGenerate": "source_producer", "ScEnsemble": "artifact_receiver", "Test": "hybrid_receiver", "Custom": "hybrid_receiver"}),
    ("HumanEval", 5): _profile("HumanEval", 5, candidates=("CustomCodeGenerate",), ensembles=(), reusable=(), binding=_humaneval_binding, signature=_humaneval_candidate_signature, source_selector=_humaneval_source_selector, candidate_projection=_humaneval_candidate_projection, supports_content_dedup=False, supports_invocation_reuse=False),
}


def get_aflow_profile(dataset: str, workflow_round: int) -> AFlowOptimizationProfile | None:
    return _PROFILES.get((str(dataset), int(workflow_round)))


def aflow_profile_names() -> tuple[str, ...]:
    return tuple(profile.profile_name for profile in _PROFILES.values())

