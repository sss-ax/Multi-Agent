"""Offline teacher-data and risk-model primitives for learned communication."""

from .schemas import (
    CounterfactualOutcome,
    CurrentAnswerLabel,
    DualRiskExample,
    FinalOutcomeLabel,
    TeacherSearchResult,
)
from .teacher_search import (
    InformationAcquisitionTeacher,
    SendCounterfactualTeacher,
)
from .risk_model import (
    DualRiskPrediction,
    build_dual_head_network,
    masked_dual_head_bce_loss,
)
from .artifacts import Phase2ArtifactWriter, assert_task_disjoint_splits, load_jsonl
from .labeling import current_candidate, label_current_answer

__all__ = [
    "CounterfactualOutcome",
    "CurrentAnswerLabel",
    "DualRiskExample",
    "FinalOutcomeLabel",
    "TeacherSearchResult",
    "InformationAcquisitionTeacher",
    "SendCounterfactualTeacher",
    "DualRiskPrediction",
    "build_dual_head_network",
    "masked_dual_head_bce_loss",
    "Phase2ArtifactWriter",
    "assert_task_disjoint_splits",
    "load_jsonl",
    "current_candidate",
    "label_current_answer",
]
