"""Command line entry point.

    triage run    --call CALL-10041            run until the human gate
    triage resume --thread <id> --approve      resume a paused run, possibly days later
    triage list                                show the sample transcripts

The split between `run` and `resume` is the demo: they are separate processes. The graph
state lives in SQLite, not in memory, so the second command picks up exactly where the
first one suspended.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triage.graph import build_graph

DATA = Path(__file__).resolve().parents[2] / "data" / "transcripts.json"
DB = Path(__file__).resolve().parents[2] / "triage.db"

console = Console()

SEVERITY_STYLE = {
    "high": "bold red",
    "medium": "yellow",
    "low": "cyan",
    "none": "green",
}


def load_transcripts() -> dict[str, dict[str, Any]]:
    return {row["call_id"]: row for row in json.loads(DATA.read_text())}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_trajectory(state: dict[str, Any]) -> None:
    table = Table(title="Trajectory", show_header=True, header_style="bold", expand=True)
    table.add_column("#", width=3, justify="right")
    table.add_column("Step")
    for i, entry in enumerate(state.get("route_log", []), 1):
        table.add_row(str(i), entry)
    console.print(table)

    tool_log = state.get("tool_log", [])
    if tool_log:
        tools = Table(title="Tool calls", show_header=False, expand=True, box=None)
        for entry in tool_log:
            style = "red" if "DEGRADED" in entry or "blocked" in entry else ""
            tools.add_row(f"  {entry}", style=style)
        console.print(tools)


def render_case(case: dict[str, Any]) -> None:
    severity = case.get("severity", "none")
    style = SEVERITY_STYLE.get(severity, "white")

    body = [
        f"[bold]{case.get('headline', '')}[/bold]",
        "",
        f"Severity:   [{style}]{severity.upper()}[/{style}]",
        f"Call:       {case.get('call_id')}",
        f"Action:     {case.get('recommended_action', '')}",
        "",
        case.get("summary", ""),
    ]
    if case.get("injection_flags"):
        body += [
            "",
            f"[bold red]Prompt-injection patterns flagged in source transcript: "
            f"{len(case['injection_flags'])}[/bold red]",
        ]
    console.print(Panel("\n".join(body), title="Case file (pending approval)", border_style=style))

    findings = case.get("findings", [])
    if not findings:
        console.print("  [green]No findings.[/green]\n")
        return

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Severity", width=9)
    table.add_column("From", width=11)
    table.add_column("Finding")
    table.add_column("Policy", width=14)
    for finding in findings:
        sev = finding.get("severity", "none")
        table.add_row(
            f"[{SEVERITY_STYLE.get(sev, 'white')}]{sev}[/{SEVERITY_STYLE.get(sev, 'white')}]",
            finding.get("source", ""),
            finding.get("summary", ""),
            finding.get("policy_ref", "") or "-",
        )
    console.print(table)


def render_meter(meter: Any) -> None:
    if not meter.calls:
        return
    table = Table(title="Telemetry", show_header=True, header_style="bold", expand=True)
    table.add_column("Node")
    table.add_column("Latency", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Output tok", justify="right")
    table.add_column("Repairs", justify="right")
    for call in meter.calls:
        table.add_row(
            call.node,
            f"{call.latency_ms:,} ms",
            f"{call.prompt_tokens:,}",
            f"{call.completion_tokens:,}",
            str(call.repairs) if call.repairs else "-",
        )
    table.add_section()
    table.add_row(
        "[bold]total[/bold]",
        f"[bold]{meter.total_ms:,} ms[/bold]",
        "",
        f"[bold]{meter.total_tokens:,}[/bold]",
        "",
    )
    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    raw = result.get("__interrupt__")
    if not raw:
        return None
    first = raw[0] if isinstance(raw, (list, tuple)) else raw
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"case": value}


def cmd_run(args: argparse.Namespace) -> int:
    transcripts = load_transcripts()
    row = transcripts.get(args.call)
    if row is None:
        console.print(f"[red]Unknown call {args.call!r}. Options: {', '.join(transcripts)}[/red]")
        return 2

    thread_id = args.thread or f"triage-{uuid.uuid4().hex[:8]}"
    graph, meter = build_graph(provider=args.provider, model=args.model, checkpoint_path=DB)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    console.print(
        Panel(
            f"call={row['call_id']}  account={row['account_id']}  scenario={row['label']}\n"
            f"thread={thread_id}  provider={args.provider}",
            title="Triage run",
            border_style="blue",
        )
    )

    with console.status("[bold blue]Running graph...", spinner="dots"):
        result = graph.invoke(
            {
                "call_id": row["call_id"],
                "account_id": row["account_id"],
                "transcript": row["transcript"],
            },
            config,
        )

    state = graph.get_state(config).values
    render_trajectory(state)
    render_meter(meter)

    payload = _interrupt_payload(result)
    if payload is None:
        console.print("[yellow]Run finished without hitting the approval gate.[/yellow]")
        return 0

    render_case(payload.get("case", {}))

    if args.auto == "pause":
        console.print(
            Panel(
                f"Run suspended and checkpointed. Resume from any process with:\n\n"
                f"  [bold]triage resume --thread {thread_id} --approve[/bold]",
                title="Awaiting human review",
                border_style="yellow",
            )
        )
        return 0

    if args.auto == "approve":
        decision = {"approved": True, "note": "auto-approved"}
    elif args.auto == "reject":
        decision = {"approved": False, "note": "auto-rejected"}
    else:
        answer = console.input("\n[bold]Approve filing this case? [y/N][/bold] ").strip().lower()
        decision = {"approved": answer in ("y", "yes"), "note": "reviewed at console"}

    return _resume(graph, config, decision)


def _resume(graph: Any, config: dict[str, Any], decision: dict[str, Any]) -> int:
    graph.invoke(Command(resume=decision), config)
    state = graph.get_state(config).values

    if not decision["approved"]:
        console.print(
            Panel("Case rejected. Nothing was written downstream.", border_style="red")
        )
        return 0

    filed = state.get("filed", {})
    status = filed.get("status", "unknown")
    if status == "offline":
        console.print(
            Panel(
                f"[yellow]{filed.get('detail')}[/yellow]\n\n"
                "Start the Java case service to persist:  [bold]make java[/bold]",
                title="Filed (degraded)",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"Case [bold]{filed.get('case_id')}[/bold] created in the case-management "
                f"service.\nstatus={status}",
                title="Filed",
                border_style="green",
            )
        )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    graph, _ = build_graph(provider=args.provider, model=args.model, checkpoint_path=DB)
    config = {"configurable": {"thread_id": args.thread}, "recursion_limit": 25}

    snapshot = graph.get_state(config)
    if not snapshot.values:
        console.print(f"[red]No checkpoint for thread {args.thread!r}.[/red]")
        return 2

    console.print(
        Panel(
            f"Restored thread [bold]{args.thread}[/bold] from checkpoint.\n"
            f"Suspended at: {snapshot.next or 'end'}",
            title="Resume",
            border_style="blue",
        )
    )
    if snapshot.values.get("case"):
        render_case(snapshot.values["case"])

    decision = {
        "approved": args.approve,
        "note": args.note or ("approved on resume" if args.approve else "rejected on resume"),
    }
    return _resume(graph, config, decision)


def cmd_list(_: argparse.Namespace) -> int:
    table = Table(title="Sample transcripts", show_header=True, header_style="bold", expand=True)
    table.add_column("Call ID")
    table.add_column("Scenario")
    table.add_column("What it exercises")
    for row in load_transcripts().values():
        table.add_row(row["call_id"], row["label"], row["note"])
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description="Multi-agent call triage")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "vllm", "stub"])
    parser.add_argument("--model", default=None, help="Override the model name")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Triage a call")
    run.add_argument("--call", default="CALL-10041")
    run.add_argument("--thread", default=None)
    run.add_argument(
        "--auto",
        choices=["ask", "approve", "reject", "pause"],
        default="ask",
        help="'pause' exits at the gate so a second process can resume it",
    )
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume", help="Resume a paused run")
    resume.add_argument("--thread", required=True)
    group = resume.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", dest="approve", action="store_false")
    resume.add_argument("--note", default="")
    resume.set_defaults(func=cmd_resume)

    listing = sub.add_parser("list", help="Show sample transcripts")
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
