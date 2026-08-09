# Agent lifecycle reference scenario

Framework-neutral reference for the agent lifecycle events proposed in
[#159](https://github.com/open-telemetry/semantic-conventions-genai/issues/159):
`gen_ai.agent.paused`, `gen_ai.agent.checkpointed`, `gen_ai.agent.resumed`,
and `gen_ai.agent.pause.resolved`.

Models a durable approval gate in front of a long-running agent, exercising
all three terminal outcomes of a pause:

- **approved** — execution resumes in a new segment (simulating a process
  restart / different worker), carrying the same `gen_ai.agent.execution.id`
  and identifying the suspended boundary via the flat
  `gen_ai.agent.resumed_from.type` / `gen_ai.agent.resumed_from.id` pair.
- **refused** — a human declines; execution terminates at the boundary. The
  refusal is a governance outcome, not an error.
- **expired** — a configured deadline passes without a decision; the producer
  emits the expiry as a record of absence. A pause with no resolution record
  stays pending; nothing is inferred from silence.

No LLM calls are involved: durable interrupt and approval mechanics with
comparable outcome categories ship in LangGraph, the OpenAI Agents SDK,
Google ADK, and Microsoft Agent Framework, and the scenario binds to the
common shape rather than any single framework's exact vocabulary.
