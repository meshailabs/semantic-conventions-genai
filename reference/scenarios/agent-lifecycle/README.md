# Agent lifecycle reference scenario (LangGraph)

Reference for the agent lifecycle events proposed in
[#159](https://github.com/open-telemetry/semantic-conventions-genai/issues/159):
`gen_ai.agent.paused`, `gen_ai.agent.checkpointed`, `gen_ai.agent.resumed`,
and `gen_ai.agent.pause.resolved`, instrumented around LangGraph's
durable-interrupt API (`interrupt()` / `Command(resume=...)` with a
checkpointer).

Every attribute is derived from LangGraph runtime state rather than
constructed by the scenario: the pause id from the `Interrupt` object the
graph returns, the execution id from the thread config, the checkpoint id
from the checkpointer's saved state (`get_state`), the resolution from the
decision delivered through `Command(resume=...)`, and the expiry deadline
from the pending interrupt's payload read back out of persisted state.

The scenario exercises all three terminal outcomes of a pause:

- **approved** — the decision arrives through `Command(resume=...)` delivered
  to a fresh graph instance over the same checkpointer, modeling resumption
  in a different worker after the original span closed. The resumed segment
  carries the same `gen_ai.agent.execution.id` and names the persisted
  checkpoint it was reconstructed from via the flat
  `gen_ai.agent.resumed_from.type` / `gen_ai.agent.resumed_from.id` pair;
  the pause linkage is carried by the resolution event.
- **refused** — the delivered decision is a decline; the graph re-enters only
  to record the decline and reach END, the gated action never continues, and
  the refusal is recorded as a governance outcome rather than an error. The
  outcome is determined by the resolution record, not by the presence or
  absence of a resumed event.
- **expired** — an expiry sweep finds the pause still pending in persisted
  state with its configured deadline passed, and emits the expiry as a record
  of absence. A pause with no resolution record stays pending; nothing is
  inferred from silence.

No LLM calls are involved: graph nodes are plain functions, so per the
reference litmus tests only framework-owned operations are emitted.

## Capture notes

Declared per the evaluation rubric, so review can weigh them explicitly:

- **Resolution semantics are gate-level, not framework-level, in LangGraph.**
  `interrupt()` payloads and `Command(resume=...)` values are
  application-defined, so generic LangGraph instrumentation cannot know that
  a given resume value means approved or refused; this scenario instruments
  at the approval-gate boundary, where those semantics are owned. Frameworks
  whose public API owns the decision (for example the OpenAI Agents SDK's
  tool-approval `approve()` / `reject()`) are the natural second reference
  for framework-derived `gen_ai.agent.pause.resolution`.
- **`thread_id` maps to `gen_ai.agent.execution.id` only when a thread hosts
  one logical execution**, as it does here. LangGraph threads can be reused
  for multiple independent runs; instrumentations on multi-run threads need
  a per-run identifier that persists across resume rather than the bare
  thread id.
- **`gen_ai.agent.id` is deliberately absent**: the registry reserves it for
  provider-assigned hosted-agent identifiers and discourages in-memory
  instance ids, and LangGraph has no such identifier. The events carry
  `gen_ai.agent.name` instead.
