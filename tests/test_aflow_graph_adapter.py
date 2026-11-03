from __future__ import annotations

import asyncio
import types

from baselines.aflow_graph_adapter import (
    AFlowGraphSession,
    make_aflow_operator_proxy_module,
)
from workflow_runtime.communication import (
    ReceiverAwareNoClosurePolicy,
    SendAllPolicy,
)


def test_aflow_output_is_compiled_into_the_shared_graph_runtime():
    session = AFlowGraphSession(
        task_id="sample-1",
        task="What is 2+2?",
        dataset="GSM8K",
        endpoints=("answer_generate", "sc_ensemble"),
        policy=SendAllPolicy(),
    )

    compiled = session.compile_output(
        operator="answer_generate",
        output={"response": "4", "reasoning": "2+2=4"},
    )

    context = session.build_receiver_context(
        sender="answer_generate",
        receiver="sc_ensemble",
        node_ids=compiled.node_ids,
    )

    assert compiled.output_kind == "candidate_answer"
    assert compiled.primary_node_id in context.sent_node_ids
    assert "What is 2+2?" in context.text
    assert "response" in context.text
    assert context.contract_satisfied
    assert session.telemetry.summary()["graph_communication_events"] == 1


def test_aflow_dependency_closure_is_reused_for_operator_context():
    session = AFlowGraphSession(
        task_id="sample-2",
        task="Generate and verify code",
        dataset="HumanEval",
        endpoints=("code_generate", "test"),
        policy=SendAllPolicy(),
    )
    candidate = session.compile_output(
        operator="code_generate",
        output={"code": "def solve(): return 1"},
    )
    verification = session.compile_output(
        operator="test",
        output={"solution": "failed", "error": "wrong answer"},
        input_node_ids=candidate.node_ids,
    )

    context = session.build_receiver_context(
        sender="test",
        receiver="code_generate",
        node_ids=verification.node_ids,
    )

    assert candidate.primary_node_id in context.closure_added_node_ids
    assert candidate.primary_node_id in context.sent_node_ids
    assert "wrong answer" in context.text


def test_aflow_no_closure_mode_reports_the_missing_dependency_boundary():
    session = AFlowGraphSession(
        task_id="sample-3",
        task="Generate and verify code",
        dataset="HumanEval",
        endpoints=("code_generate", "test"),
        policy=ReceiverAwareNoClosurePolicy(),
    )
    candidate = session.compile_output(
        operator="code_generate",
        output={"code": "def solve(): return 1"},
    )
    verification = session.compile_output(
        operator="test",
        output={"solution": "failed", "error": "wrong answer"},
        input_node_ids=candidate.node_ids,
    )

    context = session.build_receiver_context(
        sender="test",
        receiver="code_generate",
        node_ids=verification.node_ids,
    )

    assert candidate.primary_node_id not in context.sent_node_ids
    assert context.closure_added_node_ids == ()


def test_aflow_operator_proxy_preserves_return_contract_and_records_outputs():
    module = types.ModuleType("test_aflow_operator_module")

    class AnswerGenerate:
        def __init__(self, llm):
            self.llm = llm

        async def __call__(self, input, mode=None):
            return {"answer": "4", "thought": "2+2=4"}

    class ScEnsemble:
        def __init__(self, llm):
            self.llm = llm

        async def __call__(self, solutions):
            return {"response": solutions[0]}

    AnswerGenerate.__module__ = module.__name__
    ScEnsemble.__module__ = module.__name__
    module.AnswerGenerate = AnswerGenerate
    module.ScEnsemble = ScEnsemble

    session = AFlowGraphSession(
        task_id="proxy-sample",
        task="What is 2+2?",
        dataset="GSM8K",
        policy=SendAllPolicy(),
    )
    proxy = make_aflow_operator_proxy_module(
        module,
        session=session,
        communication_mode="graph_sendall",
    )

    async def run():
        answer = await proxy.AnswerGenerate(None)(input="What is 2+2?")
        ensemble = await proxy.ScEnsemble(None)(solutions=[answer["answer"]])
        return answer, ensemble

    answer, ensemble = asyncio.run(run())
    assert answer["answer"] == "4"
    assert ensemble["response"] == "4"
    assert len(session.store.snapshot().nodes) == 4
    assert session.telemetry.summary()["graph_communication_events"] == 1
