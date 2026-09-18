from __future__ import annotations

import argparse
import json
import sys

from .dataset import EvalDataset, Example
from .runner import EvalRunner, compare_to_baseline


def echo_predictor(example: Example) -> str:
    return example.expected


def _load(path: str) -> EvalDataset:
    if path.endswith(".json"):
        return EvalDataset.from_json(path)
    return EvalDataset.from_yaml(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Run a versioned LLM eval suite.")
    parser.add_argument("run")
    parser.add_argument("suite")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    dataset = _load(args.suite)
    result = EvalRunner(dataset, echo_predictor).run()
    if args.out:
        result.write(args.out)
    dumped = result.dump()
    dumped.pop("items", None)
    print(json.dumps(dumped, indent=2))
    if args.baseline:
        gate = compare_to_baseline(result, args.baseline)
        print(json.dumps({"gate": gate}, indent=2))
        if gate["regressed"]:
            print("GATE FAIL: mean score dropped beyond delta", file=sys.stderr)
            return 2
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
