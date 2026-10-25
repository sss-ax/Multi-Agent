"""Built-in TaskSpec definitions."""

from ..task_spec import (
    CODE_GENERATION_SPEC,
    MULTIHOP_SPEC,
    MULTIPLE_CHOICE_SPEC,
    NUMERIC_SPEC,
    TaskSpec,
    get_task_spec,
)

__all__ = [
    "TaskSpec",
    "NUMERIC_SPEC",
    "MULTIPLE_CHOICE_SPEC",
    "CODE_GENERATION_SPEC",
    "MULTIHOP_SPEC",
    "get_task_spec",
]
