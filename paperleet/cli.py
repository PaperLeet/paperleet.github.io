import argparse
import json
from pathlib import Path
import shutil
import sys

from paperleet.catalog import problem, problems, root, workspace_path


def _print_report(report):
    print(f"\n{report['problem']} · {'PASS' if report['passed'] else 'FAIL'}")
    for layer, (passed, total) in report["summary"].items():
        print(f"  {layer}: {passed}/{total}" if total else f"  {layer}: —")
    for test in report["tests"]:
        if not test["passed"]:
            print(f"\n  ✗ {test['id']} · {test['why']}\n{test['message']}")
    if report.get("error"):
        print(report["error"])


def _local_report(path):
    """Return None when no report exists; an unreadable report still counts as an attempt."""
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return report if isinstance(report, dict) else {}


def _list_status(entry, report):
    if report is None:
        return "Todo"
    try:
        summary = report.get("summary", {})
        complete = report.get("passed") and all(summary.get(layer, [0, 0])[0] >= count for layer, count in entry.get("tests", {}).items())
    except (AttributeError, TypeError, IndexError, KeyError):
        complete = False
    return "Accepted" if complete else "Attempted"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="paperleet", description="PaperLeet: implement and test the core algorithms of AI research papers.")
    parser.add_argument("--version", action="version", version="PaperLeet 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List problems and local progress")
    init = commands.add_parser("init", help="Copy starter into workspace")
    init.add_argument("problem")
    init.add_argument("--force", action="store_true", help="Replace an existing workspace file")
    test = commands.add_parser("test", help="Judge a Python implementation")
    test.add_argument("problem")
    test.add_argument("--impl")
    test.add_argument("--layers", default="T0,T1,T2,T3")
    test.add_argument("--report")
    mutation = commands.add_parser("mutate", help="Run the reference and all mutants against the same tests")
    mutation.add_argument("problem")
    mutation.add_argument("--report")
    ci_parser = commands.add_parser("ci", help="Verify references, mutant quality, repeatability, and timing")
    ci_parser.add_argument("--problem", action="append", help="Optionally run just one problem (repeatable)")
    ci_parser.add_argument("--repeats", type=int, default=3)
    ci_parser.add_argument("--budget", type=float, default=300, help="Total wall-clock budget for the maintained catalog (seconds)")
    viz = commands.add_parser("vizdata", help="Generate reference visualization data")
    viz.add_argument("problem", help="Problem id or 'all'")
    commands.add_parser("export", help="Generate web catalog from problem packages")
    bridge = commands.add_parser("serve", help="Start local judge bridge for the web editor")
    bridge.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            entries = problems()
            width = max([len("Problem")] + [len(entry["id"]) for entry in entries])
            print(f"{'No.':<8} {'Problem':<{width}} {'Level':<6} Status")
            for entry in entries:
                status = _list_status(entry, _local_report(root() / "reports" / (entry["id"].replace("/", "-") + ".json")))
                print(f"{entry['number']:<8} {entry['id']:<{width}} {entry['difficulty']:<6} {status}")
        elif args.command == "init":
            entry = problem(args.problem)
            destination = workspace_path(entry)
            if destination.exists() and not args.force:
                raise ValueError(f"{destination} already exists. Use --force to replace it.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry["directory"] / "starter.py", destination)
            print(destination)
        elif args.command == "test":
            from paperleet.judge.runner import run_problem
            entry = problem(args.problem)
            destination = args.report or root() / "reports" / (entry["id"].replace("/", "-") + ".json")
            report = run_problem(entry["id"], args.impl or workspace_path(entry), args.layers, destination)
            _print_report(report)
            print(f"\nReport: {destination}")
            return 0 if report["passed"] else 1
        elif args.command == "mutate":
            from paperleet.judge.mutation import mutate
            entry = problem(args.problem)
            destination = args.report or root() / "reports" / (entry["id"].replace("/", "-") + "-mutation.json")
            report = mutate(entry, destination)
            _print_report(report)
            print("\nMutant × detecting tests:")
            for mutant, killed_by in report["mutation"].items():
                print(f"  {mutant}: {', '.join(killed_by) or 'SURVIVED'}")
            for issue in report.get("quality_issues", []):
                print(issue)
            print(f"\nReport: {destination}")
            return 0 if report["passed"] else 1
        elif args.command == "ci":
            from paperleet.judge.mutation import ci
            if args.repeats < 1 or args.budget <= 0:
                raise ValueError("Repeats and budget must be positive")
            identifiers = [problem(identifier)["id"] for identifier in args.problem] if args.problem else None
            report = ci(args.repeats, identifiers, args.budget)
            print(f"{'PASS' if report['passed'] else 'FAIL'} · {report['problems']} problems · {report['tests_executed']} test cases · {report['repeats']} runs · {report['duration_s']:.2f}s")
            for identifier, mutants in report["mutation"].items():
                print(f"  {identifier}: {sum(bool(t) for t in mutants.values())}/{len(mutants)} mutants caught")
            for issue in report["issues"]:
                print(f"  ✗ {issue}")
            print(f"Report: {root() / 'reports/ci.json'}")
            return 0 if report["passed"] else 1
        elif args.command == "vizdata":
            from paperleet.export import export_catalog, vizdata
            from paperleet.browser_judge import require_browser_catalog
            require_browser_catalog(problems())
            entries = problems() if args.problem == "all" else [problem(args.problem)]
            for entry in entries:
                print(vizdata(entry))
            print(export_catalog())
        elif args.command == "export":
            from paperleet.export import export_catalog
            from paperleet.browser_judge import require_browser_catalog
            require_browser_catalog(problems())
            print(export_catalog())
        elif args.command == "serve":
            from paperleet.server import serve
            serve(args.port)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"paperleet: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
