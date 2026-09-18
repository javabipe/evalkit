from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dataset import EvalDataset, Example
from .metrics import contains, exact_match, groundedness


Predictor = Callable[[Example], str]

METRICS = {
    "exact_match": lambda pred, ex: exact_match(pred, ex.expected),
    "contains": lambda pred, ex: contains(pred, ex.expected),
    "groundedness": lambda pred, ex: groundedness(pred, ex.context),
}


@dataclass
class ItemScore:
    id: str
    score: float
    predicted: str
    latency_ms: float


@dataclass
class EvalResult:
    suite: str
    version: str
    fingerprint: str
    mean_score: float
    passed: bool
    n: int
    total_latency_ms: float
    items: list[ItemScore] = field(default_factory=list)

    def dump(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "items"},
            "items": [asdict(i) for i in self.items],
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.dump(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvalResult":
        data = json.loads(Path(path).read_text())
        items = [ItemScore(**i) for i in data.get("items", [])]
        return cls(
            suite=data["suite"],
            version=data["version"],
            fingerprint=data["fingerprint"],
            mean_score=data["mean_score"],
            passed=data["passed"],
            n=data["n"],
            total_latency_ms=data["total_latency_ms"],
            items=items,
        )


class EvalRunner:
    def __init__(self, dataset: EvalDataset, predictor: Predictor):
        self.dataset = dataset
        self.predictor = predictor
        self.metric = METRICS.get(dataset.metric, METRICS["exact_match"])

    def run(self) -> EvalResult:
        items: list[ItemScore] = []
        t0 = time.perf_counter()
        for example in self.dataset.examples:
            start = time.perf_counter()
            predicted = self.predictor(example)
            latency = (time.perf_counter() - start) * 1000
            score = self.metric(predicted, example)
            items.append(ItemScore(id=example.id, score=score, predicted=predicted, latency_ms=latency))
        mean = sum(i.score for i in items) / max(len(items), 1)
        return EvalResult(
            suite=self.dataset.name,
            version=self.dataset.version,
            fingerprint=self.dataset.fingerprint,
            mean_score=round(mean, 4),
            passed=mean >= self.dataset.min_score,
            n=len(items),
            total_latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            items=items,
        )


def compare_to_baseline(current: EvalResult, baseline_path: str | Path, delta: float = 0.02) -> dict[str, Any]:
    baseline = EvalResult.load(baseline_path)
    drop = baseline.mean_score - current.mean_score
    return {
        "baseline": baseline.mean_score,
        "current": current.mean_score,
        "drop": round(drop, 4),
        "regressed": drop > delta,
        "fingerprint_changed": current.fingerprint != baseline.fingerprint,
    }
