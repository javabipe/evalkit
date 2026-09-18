from evalkit.dataset import EvalDataset
from evalkit.runner import EvalRunner
from evalkit.metrics import groundedness


def test_fingerprint_stable(tmp_path):
    ds = EvalDataset.from_json("examples/golden_qa.json")
    assert len(ds.fingerprint) == 16
    assert ds.fingerprint == EvalDataset.from_json("examples/golden_qa.json").fingerprint


def test_runner_with_echo():
    ds = EvalDataset.from_json("examples/golden_qa.json")
    result = EvalRunner(ds, lambda ex: ex.expected).run()
    assert result.passed
    assert result.n == 3
    assert result.mean_score == 1.0


def test_groundedness_penalizes_ungrounded():
    ctx = ["We ship to Canada within five business days."]
    assert groundedness("We ship to Canada.", ctx) >= 0.5
    assert groundedness("The CEO earns twelve million dollars.", ctx) < 0.5
