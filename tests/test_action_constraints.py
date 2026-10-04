import torch

from workflow_runtime.action_constraints import ActionConstraint, build_action_constraint


class CharTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [[ord(char) for char in text]]}


def test_numeric_planner_state_selects_next_boundary():
    task = "A has 2 items and B has 3 items. What is the total?"
    first = build_action_constraint(
        role="planner",
        task_type="numeric_solve",
        missing=["missing query_spec", "missing facts", "missing plan", "missing plan_steps"],
        task_text=task,
    )
    assert first.allowed_ops == ("declare_query",)

    facts = build_action_constraint(
        role="planner",
        task_type="numeric_solve",
        missing=["missing plan", "missing plan_steps"],
        task_text=task,
        existing_logical_ids=("task", "query_spec", "facts", "fact_A", "fact_B"),
    )
    assert facts.allowed_ops == ("add_plan_step",)

    done = build_action_constraint(
        role="planner",
        task_type="numeric_solve",
        missing=[],
        task_text=task,
        existing_logical_ids=("task", "query_spec", "facts", "fact_A", "fact_B", "plan", "plan_steps"),
    )
    assert done.allowed_ops == ("done",)


def test_operation_prefix_mask_blocks_forbidden_operation_tokens():
    constraint = ActionConstraint(
        role="planner",
        task_type="numeric_solve",
        allowed_ops=("add_plan_step",),
        reason="plan is missing",
    )
    tokenizer = CharTokenizer()
    constraint.bind(tokenizer)
    logits = torch.zeros((1, 256), dtype=torch.float32)
    masked = constraint.mask_logits(logits, [])
    expected = ord("{")
    assert masked[0, expected] == 0
    assert torch.isneginf(masked[0, ord("a")])


def test_done_operation_is_constrained_to_a_complete_json_object():
    constraint = ActionConstraint(
        role="critic",
        task_type="numeric_solve",
        allowed_ops=("done",),
        reason="verification is complete",
    )
    constraint.bind(CharTokenizer())
    done_ids = [ord(char) for char in '{"op":"done"}']
    assert constraint.is_complete(done_ids)


def test_numeric_repair_is_constrained_to_revised_result_only():
    constraint = build_action_constraint(
        role="solver",
        task_type="numeric_solve",
        missing=["missing revised result"],
        task_text="A has 2 and B has 3.",
        existing_logical_ids=("task", "query_spec", "facts", "plan", "plan_steps", "calculation", "result", "verification"),
    )

    assert constraint.allowed_ops == ("set_result",)
