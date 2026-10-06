"""opendial command line: run, validate, packs, schema."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from opendial import __version__, packs
from opendial.report import write_report
from opendial.scenario import Scenario
from opendial.session import Config, find_scenarios, load_all, run_scenario


def _run(args: argparse.Namespace) -> int:
    scenarios = load_all(find_scenarios(args.paths))
    if args.tag:
        scenarios = [s for s in scenarios if set(args.tag) & set(s.tags)]
    if not scenarios:
        print("no scenarios found", file=sys.stderr)
        return 2
    config = Config(args.target, args.llm, args.tts, args.stt, args.judge, args.seed)
    outcomes = []
    for scenario in scenarios:
        outcome = asyncio.run(run_scenario(scenario, config))
        outcomes.append(outcome)
        print(f"{'PASS' if outcome.passed else 'FAIL'}  {scenario.name}")
        for r in outcome.results:
            print(f"      {r.status:<7} {r.criterion}: {r.detail}")
    index = write_report(outcomes, Path(args.out))
    passed = sum(o.passed for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} passed. Report: {index}")
    return 0 if passed == len(outcomes) else 1


def _validate(args: argparse.Namespace) -> int:
    paths = find_scenarios(args.paths)
    bad = 0
    for path in paths:
        try:
            scenario = load_all([path])[0]
            if scenario.persona.accent:
                packs.accent(scenario.persona.accent)
            if scenario.audio.line:
                packs.line(scenario.audio.line)
            print(f"ok    {path}")
        except (ValidationError, ValueError, KeyError) as e:
            bad += 1
            print(f"error {path}\n{e}")
    return 1 if bad else 0


def _packs(args: argparse.Namespace) -> int:
    for pack in packs.packs().values():
        print(f"{pack.kind:<7} {pack.id:<20} {pack.description}")
    return 0


def _schema(args: argparse.Namespace) -> int:
    print(json.dumps(Scenario.model_json_schema(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opendial", description="Test voice agents.")
    parser.add_argument("--version", action="version", version=f"opendial {__version__}")
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("run", help="place calls and check them")
    run.add_argument("paths", nargs="+", help="scenario files or directories")
    run.add_argument("--target", required=True, help="agent or transport, as module:attribute")
    run.add_argument("--llm", help="caller LLM factory, module:attribute")
    run.add_argument("--tts", help="caller TTS factory (default: fake)")
    run.add_argument("--stt", help="caller STT factory (default: fake)")
    run.add_argument("--judge", help="judge LLM factory for task_completed")
    run.add_argument("--tag", action="append", help="only run scenarios with this tag")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default=".opendial/report", help="report directory")
    run.set_defaults(func=_run)

    validate = sub.add_parser("validate", help="check scenario files")
    validate.add_argument("paths", nargs="+")
    validate.set_defaults(func=_validate)

    sub.add_parser("packs", help="list accent and line packs").set_defaults(func=_packs)
    sub.add_parser("schema", help="print the scenario JSON schema").set_defaults(func=_schema)

    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
