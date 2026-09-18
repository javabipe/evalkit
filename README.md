# evalkit

Reproducible evaluation harness for LLM systems: versioned datasets, LLM-as-judge calibration against human labels, per-run cost/latency accounting, and CI regression gates.

Built for the same workflow used on generative-commerce evals — golden sets, agreement checks, and a hard fail when quality slips.

```bash
pip install -e .
evalkit run examples/golden_qa.json
```

## What it does

- **Datasets** are JSON (YAML optional), hashed, and pinned. A silent edit to a golden item changes the suite fingerprint.
- **Judges** can be deterministic (exact / regex / contains) or model-graded. Calibration reports Cohen-style agreement against a human column.
- **Gates** fail the process when groundedness, exact-match, or cost budgets regress versus a baseline JSON.

This is the public core. Private lab packs (adversarial coding tasks, CVE reproduction scoring) stay off this repository.

## Layout

```
src/evalkit/   dataset, metrics, runner, cli
examples/      a tiny grounded-QA suite
tests/         runner + fingerprint checks
```

MIT · Paulo Moura Martins · [paulomoura.dev](https://paulomoura.dev)
