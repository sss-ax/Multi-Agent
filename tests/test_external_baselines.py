from baselines.common import (
    BaselineMessage,
    BaselineResult,
    BaselineTokenUsage,
    aggregate_external_records,
    communication_token_usage,
    count_tokens,
)
from baselines.autogen_runner import AutoGenRoundRobinRunner
from baselines.agentprune_runner import AgentPruneAutoGenLocalRunner, AgentPruneLocalPolicy
from baselines.aflow_runner import AFlowReplayRunner, AFlowSearchRunner
from scripts.evaluate_external_baselines import SingleAgentBaselineRunner, evaluate_rows


class SpaceTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": str(text).split()}


class EchoModel:
    tokenizer = SpaceTokenizer()

    def __call__(self, request):
        assert request.session_reset is True
        return "Final answer: 4"


def test_common_baseline_schema_and_token_aggregation():
    tokenizer = SpaceTokenizer()
    messages = [
        BaselineMessage(sender="solver", receiver="critic", content="check this answer"),
        BaselineMessage(sender="critic", receiver="final", content="looks good"),
    ]
    comm_tokens, comm_count = communication_token_usage(tokenizer, messages)
    usage = BaselineTokenUsage(
        input_tokens=10,
        output_tokens=3,
        communication_tokens=comm_tokens,
        communication_messages=comm_count,
        forward_calls=1,
    )
    result = BaselineResult(
        sample_id="s0",
        method="toy",
        domain="gsm8k",
        final_answer="4",
        trace=messages,
        token_usage=usage,
    )

    record = result.as_record(scores={"correct": True, "exact_match": 1.0}, gold_answer="4")
    summary = aggregate_external_records([record])

    assert count_tokens(tokenizer, "a b c") == 3
    assert record["total_model_tokens"] == 13
    assert record["communication_messages"] == 2
    assert summary["accuracy_or_pass_at_1"] == 1.0
    assert summary["communication_cost_tokens"] == comm_tokens
    assert summary["phase8_total_cost_tokens"] == comm_tokens + 13


def test_single_agent_external_evaluation_reuses_domain_scoring():
    rows = [
        {
            "sample_id": "fake_gsm8k_0",
            "domain": "gsm8k",
            "task_type": "numeric_solve",
            "question": "What is 2+2?",
            "gold_answer": "4",
        }
    ]
    runner = SingleAgentBaselineRunner(EchoModel(), EchoModel.tokenizer)

    report = evaluate_rows(
        rows=rows,
        runner=runner,
        domain="gsm8k",
        data_path="fake.jsonl",
        model_path="fake-model",
        limit=1,
        max_new_tokens=16,
    )

    assert report["execution_mode"] == "external_baseline"
    assert report["method"] == "single_agent"
    assert report["accuracy_or_pass_at_1"] == 1.0
    assert report["communication_cost_tokens"] == 0
    assert report["records"][0]["scores"]["correct"] is True


def test_autogen_roundrobin_runner_with_fake_autogen_modules(monkeypatch):
    import sys
    import types

    class FakeMessage:
        def __init__(self, source, content):
            self.source = source
            self.content = content

    class FakeResult:
        def __init__(self, messages):
            self.messages = messages

    class FakeAssistantAgent:
        def __init__(self, *, name, model_client, system_message):
            self.name = name
            self.model_client = model_client
            self.system_message = system_message

    class FakeMaxMessageTermination:
        def __init__(self, *, max_messages):
            self.max_messages = max_messages

    class FakeRoundRobinGroupChat:
        def __init__(self, agents, termination_condition):
            self.agents = agents
            self.termination_condition = termination_condition

        async def run(self, *, task):
            messages = [FakeMessage("user", task)]
            for agent in self.agents[: self.termination_condition.max_messages]:
                output = await agent.model_client.create(messages)
                content = output.content if hasattr(output, "content") else str(output)
                messages.append(FakeMessage(agent.name, content))
            return FakeResult(messages)

    agent_mod = types.ModuleType("autogen_agentchat.agents")
    agent_mod.AssistantAgent = FakeAssistantAgent
    team_mod = types.ModuleType("autogen_agentchat.teams")
    team_mod.RoundRobinGroupChat = FakeRoundRobinGroupChat
    condition_mod = types.ModuleType("autogen_agentchat.conditions")
    condition_mod.MaxMessageTermination = FakeMaxMessageTermination
    root_mod = types.ModuleType("autogen_agentchat")
    monkeypatch.setitem(sys.modules, "autogen_agentchat", root_mod)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", agent_mod)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.teams", team_mod)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.conditions", condition_mod)

    class CountingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            return "Final answer: 4" if self.calls == 4 else f"draft {self.calls}"

    row = {
        "sample_id": "fake_gsm8k_0",
        "domain": "gsm8k",
        "task_type": "numeric_solve",
        "question": "What is 2+2?",
        "gold_answer": "4",
    }
    runner = AutoGenRoundRobinRunner(CountingModel(), SpaceTokenizer())
    result = runner.run(row)

    assert result.method == "autogen_roundrobin"
    assert result.final_answer == "Final answer: 4"
    assert result.token_usage.forward_calls == 4
    assert result.token_usage.communication_messages == 5
    assert [message.sender for message in result.trace] == [
        "user",
        "solver_1",
        "solver_2",
        "critic",
        "final_solver",
    ]


def test_agentprune_local_policy_masks_stale_temporal_messages():
    tokenizer = SpaceTokenizer()
    history = [
        BaselineMessage(sender="user", receiver="group", content="task text", role="user"),
        BaselineMessage(sender="solver_1", receiver="group", content="old draft"),
        BaselineMessage(sender="solver_2", receiver="group", content="new draft"),
        BaselineMessage(sender="critic", receiver="group", content="fix this"),
    ]

    pruned = AgentPruneLocalPolicy().select(
        receiver="final_solver",
        history=history,
        tokenizer=tokenizer,
    )

    assert [message.sender for message in pruned.messages] == ["user", "solver_2", "critic"]
    assert pruned.pruned_tokens > 0
    assert any(node.sender == "solver_1" and not node.kept for node in pruned.temporal_graph)


def test_agentprune_autogen_local_runner_reports_pruning_metadata():
    class CountingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            return "Final answer: 4" if self.calls == 4 else f"draft {self.calls}"

    row = {
        "sample_id": "fake_gsm8k_0",
        "domain": "gsm8k",
        "task_type": "numeric_solve",
        "question": "What is 2+2?",
        "gold_answer": "4",
    }
    runner = AgentPruneAutoGenLocalRunner(CountingModel(), SpaceTokenizer())

    report = evaluate_rows(
        rows=[row],
        runner=runner,
        domain="gsm8k",
        data_path="fake.jsonl",
        model_path="fake-model",
        limit=1,
        max_new_tokens=16,
    )

    record = report["records"][0]
    assert report["method"] == "agentprune_autogen_local"
    assert report["accuracy_or_pass_at_1"] == 1.0
    assert record["agentprune_mode"] == "local"
    assert record["agentprune_pruned_tokens"] > 0
    assert record["full_context_equivalent_input_tokens"] >= record["input_tokens"]
    assert record["agentprune_input_saving_tokens"] > 0
    assert len(record["agentprune_temporal_graph"]) == 4


def test_aflow_replay_runner_uses_domain_workflow_and_scores():
    class CountingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            return "Final answer: 4" if self.calls == 3 else f"artifact {self.calls}"

    row = {
        "sample_id": "fake_gsm8k_0",
        "domain": "gsm8k",
        "task_type": "numeric_solve",
        "question": "What is 2+2?",
        "gold_answer": "4",
    }
    runner = AFlowReplayRunner(CountingModel(), SpaceTokenizer(), prefer_vendored=False)

    report = evaluate_rows(
        rows=[row],
        runner=runner,
        domain="gsm8k",
        data_path="fake.jsonl",
        model_path="fake-model",
        limit=1,
        max_new_tokens=16,
    )

    record = report["records"][0]
    assert report["method"] == "aflow_replay"
    assert report["accuracy_or_pass_at_1"] == 1.0
    assert record["aflow_mode"] == "local_replay"
    assert [step["name"] for step in record["aflow_workflow"]] == [
        "plan_solver",
        "consistency_checker",
        "final_numeric_answer",
    ]
    assert record["forward_calls"] == 3


def test_aflow_replay_has_mmlu_pro_adapter():
    class ChoiceModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, request):
            self.calls += 1
            return "B" if self.calls == 3 else "B seems correct"

    row = {
        "sample_id": "fake_mmlu_0",
        "domain": "mmlu_pro",
        "task_type": "multiple_choice",
        "question": "Pick one.\n\nA. wrong\nB. right\n\nAnswer with only the option letter.",
        "choices": [{"label": "A", "text": "wrong"}, {"label": "B", "text": "right"}],
        "gold_answer": "B",
    }
    runner = AFlowReplayRunner(ChoiceModel(), SpaceTokenizer())
    result = runner.run(row)

    assert result.final_answer == "B"
    assert result.metadata["aflow_workflow_key"] == "mmlu_pro"
    assert result.metadata["aflow_workflow"][-1]["name"] == "choice_finalizer"


def test_aflow_search_is_explicitly_deferred():
    runner = AFlowSearchRunner(dataset="GSM8K", max_rounds=1)
    result = runner.run({"sample_id": "s0", "domain": "gsm8k"})

    assert result.method == "aflow_search"
    assert result.metadata["aflow_search"]["status"] == "dry_run"
    assert "run.py" in result.metadata["aflow_search"]["command"]
