import torch

from workflow_runtime.action_constraints import ActionConstraint, build_action_constraint


class CharTokenizer:
    all_special_ids = ()

    def __len__(self):
        return 256

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [[ord(char) for char in text]]}

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


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


def test_v2_numeric_planner_is_constrained_to_complete_schema():
    constraint = build_action_constraint(
        role="planner",
        task_type="numeric_solve",
        missing=["missing facts", "missing plan_steps"],
        allowed_source_refs=("n1",),
        protocol_mode="semantic_delta_v2",
    )
    tokenizer = CharTokenizer()
    constraint.bind(tokenizer)
    action = (
        '{"op":"plan","facts":[{"content":"Janet lays 16 eggs",'
        '"source":"n1","quote":"16 eggs"}],"strategy":"subtract used eggs"}'
    )
    action_ids = [ord(char) for char in action]

    assert constraint.as_dict()["mode"] == "full-planner-schema"
    assert constraint.is_complete(action_ids)
    assert not constraint.is_complete(action_ids[:-1])


def test_v2_planner_schema_rejects_unknown_source_and_non_json_punctuation():
    constraint = build_action_constraint(
        role="planner",
        task_type="numeric_solve",
        missing=["missing facts", "missing plan_steps"],
        allowed_source_refs=("n1",),
        protocol_mode="semantic_delta_v2",
    )
    tokenizer = CharTokenizer()
    constraint.bind(tokenizer)

    wrong_source = (
        '{"op":"plan","facts":[{"content":"fact",'
        '"source":"n2","quote":"quote"}],"strategy":"solve"}'
    )
    assert not constraint._planner_states(ord(char) for char in wrong_source)

    fullwidth_punctuation = '{"op":"plan"，"facts":[]，"strategy":"solve"}'
    assert not constraint._planner_states(ord(char) for char in fullwidth_punctuation)


def test_v2_non_numeric_planner_forces_empty_facts_and_nonempty_strategy():
    constraint = build_action_constraint(
        role="planner",
        task_type="multiple_choice",
        missing=["missing plan_steps"],
        protocol_mode="semantic_delta_v2",
    )
    constraint.bind(CharTokenizer())

    valid = '{"op":"plan","facts":[],"strategy":"compare the options"}'
    empty_strategy = '{"op":"plan","facts":[],"strategy":""}'
    assert constraint.is_complete(ord(char) for char in valid)
    assert not constraint.is_complete(ord(char) for char in empty_strategy)


def _assert_complete_schema(constraint, action):
    constraint.bind(CharTokenizer())
    ids = [ord(char) for char in action]
    assert constraint.as_dict()["mode"] == "full-solver-schema"
    assert constraint.is_complete(ids)
    assert not constraint.is_complete(ids[:-1])


def test_v2_numeric_solver_has_complete_schema():
    constraint = build_action_constraint(
        role="solver", task_type="numeric_solve", missing=["missing result"],
        protocol_mode="semantic_delta_v2",
    )
    _assert_complete_schema(
        constraint,
        '{"op":"solve_numeric","expression":"16-3-4","value":"9"}',
    )


def test_v2_choice_solver_locks_labels_analysis_and_confidence():
    constraint = build_action_constraint(
        role="solver", task_type="multiple_choice", missing=["missing result"],
        allowed_choice_labels=("A", "B", "C", "D"), protocol_mode="semantic_delta_v2",
    )
    _assert_complete_schema(
        constraint,
            '{"op":"choose_option","answer":"D","slot_reasons":[{"slot":1,'
            '"reason":"D is supported"}],'
        '"eliminated_options":{"A":"A is contradicted"},"confidence":0.75}',
    )
    invalid = '{"op":"choose_option","answer":"Z","slot_analysis":[{"slot":1,"selected_term":"x","reason":"x"}],"eliminated_options":{"A":"x"},"confidence":0.75}'
    assert not constraint._planner_states(ord(char) for char in invalid)


def test_choice_solver_repair_forbids_the_rejected_answer():
    constraint = build_action_constraint(
        role="solver", task_type="multiple_choice", missing=["missing revised result"],
        allowed_choice_labels=("A", "B", "C"), forbidden_choice_labels=("A",),
        protocol_mode="semantic_delta_v2",
    )
    constraint.bind(CharTokenizer())
    valid = (
            '{"op":"choose_option","answer":"B","slot_reasons":[{"slot":1,'
            '"reason":"B fits"}],"eliminated_options":{"A":"wrong"},'
        '"confidence":0.75}'
    )
    rejected = valid.replace('"answer":"B"', '"answer":"A"')
    assert constraint.is_complete(ord(char) for char in valid)
    assert not constraint.is_complete(ord(char) for char in rejected)


def test_choice_solver_manifest_requires_all_four_slots():
    constraint = build_action_constraint(
        role="solver", task_type="multiple_choice", missing=["missing result"],
        allowed_choice_labels=("A", "B", "C", "D"), required_solver_slots=(1, 2, 3, 4),
        protocol_mode="semantic_delta_v2",
    )
    constraint.bind(CharTokenizer())
    action = (
            '{"op":"choose_option","answer":"A","slot_reasons":['
            '{"slot":1,"reason":"slot 1"},'
            '{"slot":2,"reason":"slot 2"},'
            '{"slot":3,"reason":"slot 3"},'
            '{"slot":4,"reason":"slot 4"}],'
        '"eliminated_options":{"B":"wrong"},"confidence":0.75}'
    )
    one_slot = action.replace(',{"slot":2,"reason":"slot 2"},', '')
    one_slot = one_slot.replace(',{"slot":3,"reason":"slot 3"},', '')
    one_slot = one_slot.replace(',{"slot":4,"reason":"slot 4"}', '')
    assert constraint.is_complete(ord(char) for char in action)
    assert not constraint.is_complete(ord(char) for char in one_slot)


def test_choice_critic_schema_is_bound_to_kernel_audit_manifest():
    constraint = build_action_constraint(
        role="critic", task_type="multiple_choice", missing=["missing verification"],
        allowed_choice_labels=("A", "B", "C", "D"),
        required_audit_slots=(1, 2), required_audit_options=("B", "C"),
        protocol_mode="semantic_delta_v2",
    )
    constraint.bind(CharTokenizer())
    complete = (
        '{"op":"audit_choice","slot_checks":[{"slot":1,"status":"claim_supported","reason":"slot 1 s1"},'
        '{"slot":2,"status":"claim_supported","reason":"slot 2 s2"}],"eliminated_checks":['
        '{"option":"B","status":"claim_supported","reason":"B b"},'
        '{"option":"C","status":"claim_supported","reason":"C c"}],'
        '"selected_option_check":{"status":"claim_supported","reason":"selected A"},"summary":"ok"}'
    )
    incomplete = complete.replace(',{"option":"C","status":"claim_supported","reason":"C c"}', '')
    assert constraint.is_complete(ord(char) for char in complete)
    assert not constraint.is_complete(ord(char) for char in incomplete)


def test_v2_code_solver_supports_json_escaped_code():
    constraint = build_action_constraint(
        role="solver", task_type="code_generation", missing=["missing code"],
        protocol_mode="semantic_delta_v2",
    )
    _assert_complete_schema(
        constraint,
        '{"op":"submit_code","code":"def solve():\\n    return \\"ok\\""}',
    )


def test_v2_evidence_solver_requires_two_visible_citations():
    constraint = build_action_constraint(
        role="solver", task_type="multihop_qa", missing=["missing result"],
        allowed_citation_refs=("sentence_source_1", "sentence_source_2", "sentence_source_3"),
        protocol_mode="semantic_delta_v2",
    )
    _assert_complete_schema(
        constraint,
        '{"op":"answer_with_evidence","answer":"yes","bridge_entity":"bridge",'
        '"citations":["sentence_source_1","sentence_source_2"],'
        '"reasoning":["first hop","second hop"]}',
    )
    one_citation = (
        '{"op":"answer_with_evidence","answer":"yes","bridge_entity":"bridge",'
        '"citations":["sentence_source_1"],"reasoning":["first hop"]}'
    )
    assert not constraint.is_complete(ord(char) for char in one_citation)


def test_v2_hotpot_solver_repair_has_complete_schema():
    constraint = ActionConstraint(
        role="solver",
        task_type="multihop_qa",
        allowed_ops=("repair_patch",),
        reason="repair",
        schema_mode="solver_repair",
        allowed_citation_refs=("sentence_source_1", "sentence_source_2"),
    )
    _assert_complete_schema(
        constraint,
        '{"op":"repair_patch","action":"PATCH_CITATIONS",'
        '"patch":{"selected_fact_ids":["sentence_source_1","sentence_source_2"],'
        '"preserve_answer":true,"reasoning_chain":["first hop","second hop"]}}',
    )


def test_v2_hotpot_solver_repair_only_exposes_kernel_selected_action_family():
    constraint = ActionConstraint(
        role="solver",
        task_type="multihop_qa",
        allowed_ops=("repair_patch",),
        reason="repair bridge only",
        schema_mode="solver_repair",
        allowed_citation_refs=("sentence_source_1", "sentence_source_2"),
        allowed_repair_actions=("PATCH_BRIDGE",),
    )
    constraint.bind(CharTokenizer())
    bridge = (
        '{"op":"repair_patch","action":"PATCH_BRIDGE",'
        '"patch":{"bridge_entity":"Paris","preserve_answer":true,'
        '"reasoning_chain":["repair the bridge"]}}'
    )
    citation = (
        '{"op":"repair_patch","action":"PATCH_CITATIONS",'
        '"patch":{"selected_fact_ids":["sentence_source_1","sentence_source_2"],'
        '"preserve_answer":true,"reasoning_chain":["first","second"]}}'
    )
    assert constraint.is_complete(ord(char) for char in bridge)
    assert not constraint.is_complete(ord(char) for char in citation)


def _critic_constraint():
    return build_action_constraint(
        role="critic",
        task_type="numeric_solve",
        missing=["missing verification"],
        protocol_mode="semantic_delta_v2",
    )


def test_v2_critic_verified_has_complete_schema():
    constraint = _critic_constraint()
    constraint.bind(CharTokenizer())
    action = (
        '{"op":"judge_numeric","model_verdict":"consistent","error_type":"none",'
        '"error_location":"","reason":"calculation is correct",'
        '"repair_instruction":""}'
    )
    ids = [ord(char) for char in action]

    assert constraint.as_dict()["mode"] == "full-critic-schema"
    assert constraint.is_complete(ids)
    assert not constraint.is_complete(ids[:-1])


def test_v2_critic_need_fix_requires_complete_nonempty_diagnosis():
    constraint = _critic_constraint()
    constraint.bind(CharTokenizer())
    valid = (
        '{"op":"judge_numeric","model_verdict":"inconsistent","error_type":"arithmetic",'
        '"error_location":"expression","reason":"2 plus 3 is not 6",'
        '"repair_instruction":"recompute the sum"}'
    )
    empty_error_type = (
        '{"op":"judge_numeric","model_verdict":"inconsistent","error_type":"none",'
        '"error_location":"expression","reason":"wrong",'
        '"repair_instruction":"recompute"}'
    )
    empty_location = (
        '{"op":"judge_numeric","model_verdict":"inconsistent","error_type":"arithmetic",'
        '"error_location":"","reason":"wrong",'
        '"repair_instruction":"recompute"}'
    )
    empty_instruction = (
        '{"op":"judge_numeric","model_verdict":"inconsistent","error_type":"arithmetic",'
        '"error_location":"expression","reason":"wrong",'
        '"repair_instruction":""}'
    )

    assert constraint.is_complete(ord(char) for char in valid)
    assert not constraint.is_complete(ord(char) for char in empty_error_type)
    assert not constraint.is_complete(ord(char) for char in empty_location)
    assert not constraint.is_complete(ord(char) for char in empty_instruction)


def test_v2_critic_rejects_legacy_verify_and_noncanonical_shape():
    constraint = _critic_constraint()
    constraint.bind(CharTokenizer())
    legacy = (
        '{"op":"verify","target":"result","status":"verified",'
        '"error_type":"","error_location":"","reason":"",'
        '"repair_instruction":"","preserve":[],"requested_fragments":[]}'
    )
    extra_field = (
        '{"op":"judge_numeric","model_verdict":"consistent","error_type":"none",'
        '"error_location":"","reason":"correct","repair_instruction":"",'
        '"preserve":[]}'
    )
    wrong_status = (
        '{"op":"judge_numeric","model_verdict":"ok","error_type":"none",'
        '"error_location":"","reason":"correct","repair_instruction":""}'
    )

    assert not constraint.is_complete(ord(char) for char in legacy)
    assert not constraint.is_complete(ord(char) for char in extra_field)
    assert not constraint.is_complete(ord(char) for char in wrong_status)


def test_v2_code_critic_forces_bounded_diagnosis_to_close():
    constraint = build_action_constraint(
        role="critic", task_type="code_generation", missing=["missing verification"],
        protocol_mode="semantic_delta_v2",
    )
    constraint.bind(CharTokenizer())
    valid = '{"op":"diagnose_code","diagnosis":"syntax error","repair_instruction":"fix line one"}'
    too_long = '{"op":"diagnose_code","diagnosis":"' + ("x" * 65) + '","repair_instruction":"fix"}'
    assert constraint.is_complete(ord(char) for char in valid)
    assert not constraint.is_complete(ord(char) for char in too_long)


def test_v2_choice_and_evidence_critic_use_domain_schemas():
    choice = build_action_constraint(
        role="critic", task_type="multiple_choice", missing=["missing verification"],
        allowed_choice_labels=("A", "B"), protocol_mode="semantic_delta_v2",
    )
    choice.bind(CharTokenizer())
    assert choice.is_complete(ord(char) for char in (
        '{"op":"audit_choice","slot_checks":[{"slot":1,"status":"supported","reason":"B fits"}],'
        '"eliminated_checks":[{"option":"A","status":"contradicted","reason":"A conflicts"}],'
        '"selected_option_check":{"status":"supported","reason":"B is supported"},'
        '"summary":"The candidate claims were checked."}'
    ))

    evidence = build_action_constraint(
        role="critic", task_type="multihop_qa", missing=["missing verification"],
        protocol_mode="semantic_delta_v2",
    )
    evidence.bind(CharTokenizer())
    assert evidence.is_complete(ord(char) for char in (
        '{"op":"judge_evidence","answer_support":"supported","bridge_status":"valid",'
        '"citation_status":"valid","repair_target":"none","reason":"both hops support the answer"}'
    ))
