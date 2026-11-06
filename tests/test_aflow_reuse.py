from __future__ import annotations

import asyncio
import types

import pytest

from baselines.aflow_graph_adapter import AFlowGraphSession, make_aflow_operator_proxy_module
from baselines.aflow_reuse import AFlowInvocationKey, AFlowInvocationReuse


class SpaceTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": str(text).split()}


def test_exact_duplicate_outputs_share_one_canonical_artifact():
    session = AFlowGraphSession(
        task_id="task-1",
        task="2+3",
        dataset="GSM8K",
        content_dedup=True,
        tokenizer=SpaceTokenizer(),
    )

    first = session.compile_output(operator="Solver", output="calculation: 2+3=5")
    second = session.compile_output(operator="Solver", output="calculation: 2+3=5")

    assert first.primary_node_id == second.primary_node_id
    assert second.reused is True
    assert session.candidate_pool_ids() == (first.primary_node_id,)
    assert len(session.candidate_occurrence_ids()) == 2
    assert session.artifact_multiplicity(first.primary_node_id) == 2
    nodes = session.store.snapshot().nodes
    assert sum(node.type == "aflow_artifact" for node in nodes.values()) == 1
    assert sum(node.type == "aflow_invocation" for node in nodes.values()) == 2


def test_different_outputs_create_distinct_artifacts():
    session = AFlowGraphSession(
        task_id="task-1",
        task="2+3",
        content_dedup=True,
        tokenizer=SpaceTokenizer(),
    )
    first = session.compile_output(operator="Solver", output="5")
    second = session.compile_output(operator="Solver", output="6")
    assert first.primary_node_id != second.primary_node_id
    assert len(session.candidate_pool_ids()) == 2


def test_singleflight_executes_concurrent_duplicate_once():
    reuse = AFlowInvocationReuse()
    key = AFlowInvocationKey.build(
        task_id="task-1",
        operator="Solver",
        args=(),
        kwargs={"problem": "2+3"},
        model_fingerprint="model",
        generation_config={"temperature": 0},
    )
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "5"

    async def run():
        return await asyncio.gather(
            *(reuse.get_or_compute(key, compute) for _ in range(5))
        )

    results = asyncio.run(run())
    assert calls == 1
    assert [item.value for item in results] == ["5"] * 5
    assert sum(item.hit for item in results) == 4


def test_singleflight_cleans_inflight_after_failure():
    reuse = AFlowInvocationReuse()
    key = AFlowInvocationKey.build(
        task_id="task-1",
        operator="Solver",
        args=(),
        kwargs={"problem": "bad"},
        model_fingerprint="model",
        generation_config={"temperature": 0},
    )

    async def broken():
        raise RuntimeError("generation failed")

    async def run():
        await reuse.get_or_compute(key, broken)

    with pytest.raises(RuntimeError, match="generation failed"):
        asyncio.run(run())
    assert not reuse._inflight


def test_graph_shadow_path_does_not_replace_original_input():
    class Operator:
        __module__ = "test_operator_module"

        async def __call__(self, *, input):
            return input

    module = types.ModuleType("test_operator_module")
    module.Operator = Operator
    session = AFlowGraphSession(task_id="task-1", task="question", content_dedup=True)
    proxy = make_aflow_operator_proxy_module(
        module,
        session=session,
        communication_mode="graph_shadow",
    )
    result = asyncio.run(proxy.Operator()(input="original"))
    assert result == "original"


def test_candidate_pool_is_consumed_after_ensemble():
    session = AFlowGraphSession(
        task_id="task-1",
        task="2+3",
        content_dedup=True,
        tokenizer=SpaceTokenizer(),
    )
    session.compile_output(operator="Custom", output="5")
    session.compile_output(operator="Custom", output="5")
    assert len(session.candidate_pool_ids()) == 1
    session.consume_candidate_pool()
    assert session.candidate_pool_ids() == ()
    assert session.candidate_occurrence_ids() == ()


def test_internal_invocations_are_not_in_receiver_context():
    session = AFlowGraphSession(
        task_id="task-1",
        task="2+3",
        content_dedup=True,
        tokenizer=SpaceTokenizer(),
    )
    compiled = session.compile_output(operator="Custom", output="5")
    context = session.build_receiver_context(
        sender="Custom",
        receiver="ScEnsemble",
        node_ids=(compiled.primary_node_id,),
    )
    assert compiled.invocation_node_id not in context.sent_node_ids


def test_proxy_singleflight_reuses_real_operator_execution():
    calls = []

    class Operator:
        __module__ = "test_operator_module"

        async def __call__(self, *, input):
            calls.append(input)
            await asyncio.sleep(0.01)
            return "same answer"

    module = types.ModuleType("test_operator_module")
    module.Operator = Operator
    session = AFlowGraphSession(
        task_id="task-1",
        task="question",
        content_dedup=True,
        invocation_reuse=True,
        model_fingerprint="model",
        generation_config={"do_sample": False, "temperature": 0},
        tokenizer=SpaceTokenizer(),
    )
    proxy = make_aflow_operator_proxy_module(
        module,
        session=session,
        communication_mode="graph_sendall",
    )

    async def run():
        return await asyncio.gather(*(proxy.Operator()(input="same") for _ in range(5)))

    assert asyncio.run(run()) == ["same answer"] * 5
    assert calls == ["same"]
    summary = session.telemetry.summary()
    assert summary["aflow_invocation_reuse_hits"] == 4
