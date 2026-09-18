from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Example:
    id: str
    input: str
    expected: str
    context: list[str] = field(default_factory=list)
    human_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "context": self.context,
            "human_score": self.human_score,
            "metadata": self.metadata,
        }


@dataclass
class EvalDataset:
    name: str
    version: str
    examples: list[Example]
    metric: str = "exact_match"
    cost_budget_usd: float | None = None
    min_score: float = 0.8

    @property
    def fingerprint(self) -> str:
        payload = json.dumps([ex.dump() for ex in self.examples], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(f"{self.name}:{self.version}:{payload}".encode()).hexdigest()[:16]

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        data = json.loads(Path(path).read_text())
        examples = [Example(**ex) for ex in data["examples"]]
        return cls(
            name=data["name"],
            version=data["version"],
            metric=data.get("metric", "exact_match"),
            examples=examples,
            cost_budget_usd=data.get("cost_budget_usd"),
            min_score=data.get("min_score", 0.8),
        )

    # YAML is accepted as a thin JSON-compatible subset via stdlib-only loader.
    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalDataset":
        text = Path(path).read_text()
        if text.lstrip().startswith("{"):
            return cls.from_json(path)
        return cls.from_json(_yaml_to_json_file(path))


def _yaml_to_json_file(path: str | Path) -> Path:
    """Minimal indented YAML (no tags) → dict. Enough for golden suites."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(Path(path).read_text())
        tmp = Path(path).with_suffix(".generated.json")
        tmp.write_text(json.dumps(data))
        return tmp
    except Exception:
        pass
    # Fallback: the examples ship as JSON; YAML path is optional.
    json_path = Path(path).with_suffix(".json")
    if json_path.exists():
        return json_path
    raise RuntimeError("Install pyyaml or use a .json suite")
