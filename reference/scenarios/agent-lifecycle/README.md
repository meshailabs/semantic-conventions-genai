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
  carries the same `gen_ai.agent.execution.id` and identifies the suspended
  boundary via the flat `gen_ai.agent.resumed_from.type` /
  `gen_ai.agent.resumed_from.id` pair.
- **refused** — the delivered decision is a decline; the gated action never
  continues, no `gen_ai.agent.resumed` event is emitted, and the refusal is
  recorded as a governance outcome rather than an error.
- **expired** — an expiry sweep finds the pause still pending in persisted
  state with its configured deadline passed, and emits the expiry as a record
  of absence. A pause with no resolution record stays pending; nothing is
  inferred from silence.

No LLM calls are involved: graph nodes are plain functions, so per the
reference litmus tests only framework-owned operations are emitted.
