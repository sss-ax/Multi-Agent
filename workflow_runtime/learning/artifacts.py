"""JSONL persistence and task-level split leakage audits for Phase 2 data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .error_features import assert_deployable_features
from .schemas import DualRiskExample, TeacherSearchResult


class Phase2ArtifactWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.risk_path = self.output_dir / "dual_risk_examples.jsonl"
        self.teacher_path = self.output_dir / "communication_teacher_examples.jsonl"

    def write_risk(self, example: DualRiskExample) -> None:
        assert_deployable_features(example.state_features)
        assert_deployable_features(example.policy_features)
        self._append(self.risk_path, example.as_dict())

    def write_teacher(self, result: TeacherSearchResult) -> None:
        self._append(self.teacher_path, result.as_dict())

    @staticmethod
    def _append(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    return tuple(
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def assert_task_disjoint_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
) -> None:
    """Reject splitting counterfactual branches of one task across datasets."""
    task_to_split: dict[str, str] = {}
    collisions: list[str] = []
    for split_name, rows in splits.items():
        for row in rows:
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                raise ValueError(f"{split_name} row is missing task_id")
            previous = task_to_split.setdefault(task_id, str(split_name))
            if previous != str(split_name):
                collisions.append(f"{task_id}:{previous}/{split_name}")
    if collisions:
        raise ValueError(f"task leakage across splits: {sorted(set(collisions))}")
