# Architecture

Six views of the same system, ordered from structure to consequence. The first three
describe what the code does; the last three describe why it is shaped that way.

- [1. Graph topology](#1-graph-topology) — the static shape
- [2. Anatomy of a run](#2-anatomy-of-a-run) — what actually happens, in order
- [3. State accumulation](#3-state-accumulation) — how the graph remembers
- [4. Suspend and resume](#4-suspend-and-resume) — why the human gate is durable
- [5. Trust boundary](#5-trust-boundary) — what the model can and cannot reach
- [6. Production shape](#6-production-shape) — what scaling this looks like

---

## 1. Graph topology

A supervisor dispatches specialists, each returns to the supervisor, and the loop exits
into a human approval gate. The cycle is the reason this is a graph and not a chain — and
the reason termination has to be enforced rather than requested.

```
                              START
                                │
                                ▼
                ┌───────────────────────────────┐
                │ intake                        │
                │  • scan for injection patterns│
                │  • steps = 0                  │
                └───────────────┬───────────────┘
                                ▼
             ┌─────────────────────────────────────┐
     ┌──────▶│ supervisor                          │
     │       │  • routes, never analyses           │
     │       │  • GUARD: step cap + completed-set  │
     │       └─────────────────┬───────────────────┘
     │                         │
     │          conditional edge on state["next"]
     │                         │
     │      ┌──────────┬───────┴───────┬──────────────┐
     │      ▼          ▼               ▼              ▼
     │ ┌─────────┐ ┌──────────┐ ┌────────────┐  ┌──────────┐
     │ │compliance│ │sentiment │ │ resolution │  │ finalize │
     │ │ search_  │ │ get_acct │ │  search_   │  │ compute  │
     │ │ policy   │ │ _context │ │  policy    │  │ severity │
     │ └────┬─────┘ └────┬─────┘ └─────┬──────┘  └────┬─────┘
     │      │            │             │              │
     └──────┴────────────┴─────────────┘              │
              (each returns to supervisor)            │
                                                      ▼
                                        ╔═════════════════════════╗
                                        ║  approval               ║
                                        ║  interrupt() ── SUSPEND ║
                                        ╚════════════┬════════════╝
                                                     │
                                    ┌────────────────┴────────────────┐
                                 rejected                         approved
                                    │                                 │
                                    ▼                                 ▼
                                  END                          ┌─────────────┐
                            (nothing written)                  │  file_case  │
                                                               │  → Java svc │
                                                               └──────┬──────┘
                                                                      ▼
                                                                     END
```

**Why the supervisor doesn't analyse.** Keeping domain reasoning out of the router means a
specialist can be swapped, retuned, or replaced without touching routing behaviour. It also
keeps the router's prompt short, which keeps its latency and its failure surface small.

**Why the guard is in code.** The supervisor gets a vote on what runs next, not a veto on
stopping. A completed-set check and a hard step cap sit above it in `graph.py`. Prompts
express intent; the graph enforces it.

See [`src/triage/graph.py`](../src/triage/graph.py).

---

## 2. Anatomy of a run

One transcript, start to suspension. Note the density of model calls — this is what makes
the serving tier an architectural concern rather than a deployment detail.

```
Worker    Graph    Super   Spec    Tools    LLM     CP
  │         │        │       │       │       │       │
  ├─invoke─▶│        │       │       │       │       │
  │         ├─intake─────────────────────────────────▶│ save
  │         │        │       │       │       │       │
  │         ├───────▶│       │       │       │       │
  │         │        ├───────────────────────▶│      │   "who's next?"
  │         │        │◀──────────────────────┤│      │   RouteDecision
  │         │        │  ▲                    │       │
  │         │        │  └── code overrides if illegal │
  │         │        ├──────▶│       │       │       │
  │         │        │       ├──────────────▶│       │   bind_tools
  │         │        │       │◀──────────────┤       │   tool_calls[]
  │         │        │       ├──────▶│       │       │
  │         │        │       │◀──────┤       │       │   policy text
  │         │        │       ├──────────────▶│       │   + observations
  │         │        │       │◀──────────────┤       │   ComplianceReport
  │         │        │       │       │       │       │
  │         │◀───────────────┤ findings += [...]     │
  │         ├───────────────────────────────────────▶│ save
  │         │        │       │       │       │       │
  │         │   ┌────┴───────────────────────────────────────┐
  │         │   │  repeat for sentiment, resolution          │
  │         │   └────┬───────────────────────────────────────┘
  │         │        │       │       │       │       │
  │         ├─finalize───────────────────────▶│      │   CaseFile
  │         │  severity = max(findings) ◀── computed, not generated
  │         ├───────────────────────────────────────▶│ save
  │         │        │       │       │       │       │
  │◀─ __interrupt__ ─┤  SUSPENDED                    │
  │         │        │       │       │       │       │
```

**Seven model calls, one shared prefix.** Every specialist call re-sends the same fenced
transcript and the same tool schemas. Without prefix caching, that prefix is re-processed
from scratch seven times — which is why a 7B on a CPU couldn't finish a run in twenty
minutes, and why `--enable-prefix-caching` is the flag that matters most for agent traffic.

**Severity is computed, not generated.** The model writes the prose summary; `finalize`
takes the max severity across findings in code. Anything downstream systems route or alert
on should be derived deterministically from evidence.

---

## 3. State accumulation

Nodes don't mutate shared state — they return updates that LangGraph merges through
reducers declared on the state schema. Annotated keys append; plain keys overwrite.

```
  key              reducer        intake  compl.  sent.  resol.  final
  ─────────────────────────────────────────────────────────────────────
  transcript       (set once)      ████    ████    ████   ████    ████
  injection_flags  overwrite       ██──    ────    ────   ────    ────
  findings         operator.add     ──     ██──▶   ██──▶  ──▶     ═══▶
  completed        operator.add     ──     [c]▶    [c,s]▶ [c,s,r]▶
  route_log        operator.add    ██▶     ──▶     ──▶    ──▶     ══▶
  next             overwrite        ──     compl.  sent.  resol.  finalize
  steps            overwrite        0      1       2      3       4
  case             overwrite        ──     ────    ────   ────    ████
```

**The bug this schema caused.** `next` was originally not declared in `TriageState`.
LangGraph silently drops keys that aren't in the schema — no error, no warning — so every
routing decision evaporated and the conditional edge fell through to its default branch.
No specialist ever ran, and the system still produced a confident, well-formatted, empty
case file.

That is the characteristic failure mode of agent systems in one bug: **the failure surface
is plausible output, not a stack trace.** It is why the tests assert on trajectory — which
nodes actually executed — rather than only on the final answer.

---

## 4. Suspend and resume

`interrupt()` doesn't block a thread. It persists the run and returns. A different process,
hours later, resumes mid-node from the checkpoint.

```
  PROCESS A  (worker pod, 14:02)          PROCESS B  (reviewer, 16:40)
        │                                          │
        │ invoke(transcript)                       │
        ├──────────────┐                           │
        │   graph runs │                           │
        │◀─────────────┘                           │
        │                                          │
        ├────────────────────┐                     │
        │  interrupt()       │                     │
        │  ↓                 ▼                     │
        │             ┌─────────────┐              │
        │             │  SQLite /   │              │
        │             │  Postgres   │              │
        │             │  thread_id  │              │
        │             │  = CALL-…   │              │
        │             └─────────────┘              │
        │                    ▲                     │
        │  process exits ✝   │                     │
        ✝                    │                     │
                             │   get_state(thread) │
                             │◀────────────────────┤
                             │                     │
                             │  Command(resume=    │
                             │    {approved:true}) │
                             │◀────────────────────┤
                             └─────────┐           │
                                       ▼           │
                                  resumes MID-NODE │
                                       │           │
                                       ├──────────▶│ POST /cases
                                       │           ├────────────▶ Java
                                       │           │◀─ CASE-1002 ─┘
```

Try it — these are two separate processes:

```bash
make pause                                 # suspends at the gate, exits
triage resume --thread demo --approve      # a different process finishes it
```

**Why this matters operationally.** A suspended run occupies zero workers. State lives in
the checkpointer, not in a pod's memory, so a two-hour human review costs nothing to hold
open and resume can land on any pod — including one that didn't exist when the run started.
The same mechanism gives you crash recovery for free: a pod dying mid-run resumes from its
last checkpoint.

The corollary is that **work can replay**, which is exactly why the Java service dedupes on
`call_id`. Without idempotency at the write boundary, a pod restart files duplicate
compliance cases.

---

## 5. Trust boundary

The most important diagram here. Everything above is about making the system work;
this is about bounding what happens when it doesn't.

```
   ┌─── UNTRUSTED ────────────────────────────────────────┐
   │  transcript  ← third-party text, same threat class   │
   │              as a retrieved doc or scraped page      │
   └──────────────────────┬───────────────────────────────┘
                          │  fenced + scanned at intake
                          ▼
   ┌─── what the MODEL can reach ─────────────────────────┐
   │                                                      │
   │    ✓ search_policy(query)          read-only         │
   │    ✓ get_account_context(id)       read-only, PII    │
   │                                     redacted at the  │
   │                                     tool boundary    │
   │                                                      │
   │    ✗ file_case()   ── NOT IN ANY TOOL LIST ──        │
   │                                                      │
   └──────────────────────────────────────────────────────┘
                          │
                    ╔═════▼══════╗
                    ║   HUMAN    ║   ← the only path to a write
                    ╚═════┬══════╝
                          ▼
              ┌───────────────────────┐
              │  Java case service    │
              │  • revalidates payload│  ← trusts nothing upstream
              │  • dedupes on call_id │  ← replay-safe
              └───────────────────────┘
```

**The model has no path to the write action.** Not "is instructed not to" — *cannot*.
`file_case` is a plain function the graph calls after approval; it is absent from every
tool list, so it cannot be selected. A fully successful prompt injection reaches only two
read-only tools, and the case still requires a human.

This is the structural answer to excessive agency: you do not ask a model to be disciplined
about destructive actions, you remove the capability.

**Detection is a tripwire, not a control.** `scan_for_injection` flags instruction-shaped
spans and the fencing lowers the hit rate, but neither is load-bearing. `CALL-10043` carries
a live payload instructing the system to report the call clean; on a live run it was flagged
and reported as a finding rather than obeyed, and the verdict stayed HIGH. That outcome is
asserted in [`tests/test_evals.py`](../tests/test_evals.py).

**PII is redacted at the tool boundary, not in the prompt.** Anything that reaches a context
window should be assumed logged, cached, and potentially echoed back.

---

## 6. Production shape

LangGraph is a library, not a runtime. It does not manage instances — `graph.invoke()` is an
ordinary function call in your process. Scaling this is the same problem as scaling any
stateless service.

```
                                            ┌──────────────────┐
  recordings ──▶ ┌─────────┐  ┌──────────┐  │  vLLM replicas   │
                 │  Kafka  │─▶│ triage   │─▶│  (GPU)           │
                 │  / SQS  │  │ workers  │  │                  │
                 └─────────┘  │  × N     │◀─│  ← THE BOTTLENECK│
                              └────┬─────┘  └──────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   Postgres      │  writes after every
                          │   checkpointer  │  super-step — hot path
                          └────────┬────────┘
                                   │
                                   ▼
                        ┌────────────────────┐
                        │ human review queue │  runs suspended here
                        │  (0 workers held)  │  cost nothing to hold
                        └─────────┬──────────┘
                                  ▼
                          Java case service
```

The graph object is compiled **once** at process start and shared across every run. It is
effectively stateless — all mutable per-run state lives in the checkpointer, keyed by
`thread_id`. That is what lets one object serve thousands of concurrent runs.

**Concurrency comes from async, not cores.** A run is ~7 HTTP calls to the inference tier
and a few tens of milliseconds of actual compute — roughly a 1000:1 ratio of waiting to
working. `await graph.ainvoke(...)` behind a semaphore gives a single core hundreds of
in-flight runs. A `for` loop calling `invoke()` gives you exactly one, on any hardware.

**Scale on queue depth, not CPU.** Workers are almost entirely idle. Sizing pods on cores
buys idle CPU; the constrained resource is GPU capacity at the inference tier, and
unbounded worker concurrency just relocates the queue onto the GPU where it degrades worse.
The semaphore is backpressure, not decoration.
