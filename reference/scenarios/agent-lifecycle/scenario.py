"""Reference implementation for agent lifecycle events (#159), on LangGraph.

Instruments LangGraph's durable-interrupt API: `interrupt()` pauses a graph at
a human-approval gate, a checkpointer persists state, and `Command(resume=...)`
delivers the decision to a later invocation, possibly in a different process.
Every lifecycle attribute is derived from LangGraph runtime state: the pause id
from the returned `Interrupt` object, the execution id from the thread config,
the checkpoint id from the checkpointer's saved state, and the resolution from
the decision delivered through `Command`. No LLM calls are involved; graph
nodes are plain functions, so per the litmus tests only framework-owned
operations are emitted.
"""

from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry.trace import SpanKind
from reference_shared import flush_and_shutdown, reference_event_logger, reference_tracer, setup_otel

_reference_tracer = reference_tracer()
_LIFECYCLE_LOGGER = "gen_ai.agent.lifecycle.reference"

AGENT_NAME = "payment_reconciler"


class GateState(TypedDict, total=False):
    action: str
    approved: bool
    result: str


def gated_action(state: GateState) -> GateState:
    """Perform a high-impact action only after an out-of-band human decision.

    `interrupt()` suspends the graph here; the value it returns on a later
    invocation is whatever `Command(resume=...)` delivered.
    """
    decision = interrupt(
        {
            "action": state["action"],
            "reason": "human_input",
            "deadline": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
        }
    )
    if decision["approved"]:
        return {"approved": True, "result": f"{state['action']}: done"}
    return {"approved": False, "result": f"{state['action']}: declined"}


def build_graph(checkpointer: InMemorySaver):
    builder = StateGraph(GateState)
    builder.add_node("gated_action", gated_action)
    builder.add_edge(START, "gated_action")
    builder.add_edge("gated_action", END)
    return builder.compile(checkpointer=checkpointer)


def _emit(event_name: str, attributes: dict) -> None:
    reference_event_logger(_LIFECYCLE_LOGGER).emit(
        event_name=event_name,
        body=event_name.rsplit(".", 1)[-1],
        attributes=attributes,
    )


def _invoke_agent_span():
    span = _reference_tracer.start_as_current_span(
        f"invoke_agent {AGENT_NAME}", kind=SpanKind.INTERNAL
    )
    return span


def _pause_until_interrupt(graph, config, action: str):
    """Run the graph to its interrupt and emit paused + checkpointed.

    Returns the pause id from the Interrupt object LangGraph returns and the
    checkpoint id from the checkpointer's saved state.
    """
    thread_id = config["configurable"]["thread_id"]
    result = graph.invoke({"action": action}, config)
    (intr,) = result["__interrupt__"]
    pause_id = intr.id
    _emit(
        "gen_ai.agent.paused",
        {
            "gen_ai.agent.execution.id": thread_id,
            "gen_ai.agent.pause.reason": intr.value["reason"],
            "gen_ai.agent.pause.id": pause_id,
            "gen_ai.agent.name": AGENT_NAME,
        },
    )
    # The checkpointer persisted the interrupted state; the saved checkpoint id
    # comes from the thread's runtime state, not from anything we invented.
    saved = graph.get_state(config)
    checkpoint_id = saved.config["configurable"]["checkpoint_id"]
    _emit(
        "gen_ai.agent.checkpointed",
        {
            "gen_ai.agent.execution.id": thread_id,
            "gen_ai.agent.checkpoint.id": checkpoint_id,
            "gen_ai.agent.name": AGENT_NAME,
        },
    )
    return pause_id, checkpoint_id


def run_approved_flow(checkpointer: InMemorySaver):
    """Pause at the gate, approve out of band, resume in a new segment.

    Each flow dedicates one LangGraph thread to one logical execution, so the
    thread id satisfies the execution-id uniqueness requirement here; see the
    README capture notes for the multi-run-thread caveat.
    """
    print("  [lifecycle] approved: interrupt -> checkpoint -> Command(resume) in new segment")
    config = {"configurable": {"thread_id": "exec-approved-01"}}
    thread_id = config["configurable"]["thread_id"]

    with _invoke_agent_span() as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("gen_ai.agent.execution.id", thread_id)
        pause_id, checkpoint_id = _pause_until_interrupt(
            build_graph(checkpointer), config, "release payment batch"
        )
    # Segment 1's span is closed; the hosting process could exit here.

    # A different graph instance over the same checkpointer models a restart
    # in another worker. The human decision arrives through Command(resume=...).
    graph2 = build_graph(checkpointer)
    with _invoke_agent_span() as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("gen_ai.agent.execution.id", thread_id)
        decision = {"approved": True}
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": thread_id,
                "gen_ai.agent.pause.id": pause_id,
                "gen_ai.agent.pause.resolution": "approved" if decision["approved"] else "refused",
                "gen_ai.agent.name": AGENT_NAME,
            },
        )
        # The new worker reconstructs state from the persisted checkpoint, so
        # the resumed segment names the CHECKPOINT it continues from; the
        # pause linkage is already carried by the resolution event above.
        _emit(
            "gen_ai.agent.resumed",
            {
                "gen_ai.agent.execution.id": thread_id,
                "gen_ai.agent.resumed_from.type": "checkpoint",
                "gen_ai.agent.resumed_from.id": checkpoint_id,
                "gen_ai.agent.name": AGENT_NAME,
            },
        )
        outcome = graph2.invoke(Command(resume=decision), config)
        print("    ->", outcome["result"])


def run_refused_flow(checkpointer: InMemorySaver):
    """Pause at the gate; the human declines. No resumed segment follows."""
    print("  [lifecycle] refused: interrupt -> Command(resume=declined) -> terminates at boundary")
    config = {"configurable": {"thread_id": "exec-refused-01"}}
    thread_id = config["configurable"]["thread_id"]

    with _invoke_agent_span() as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("gen_ai.agent.execution.id", thread_id)
        graph = build_graph(checkpointer)
        pause_id, _checkpoint_id = _pause_until_interrupt(graph, config, "drop staging database")

        decision = {"approved": False}
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": thread_id,
                "gen_ai.agent.pause.id": pause_id,
                "gen_ai.agent.pause.resolution": "approved" if decision["approved"] else "refused",
                "gen_ai.agent.name": AGENT_NAME,
            },
        )
        # Delivering the refusal re-enters the graph so it can record the
        # decline and reach END; the gated action itself never continues.
        # Continuation happens within this same still-open span, and per the
        # conventions instrumentations MAY emit a resumed event for such a
        # decline-path segment; this scenario omits it, and the outcome is
        # determined by the resolution record above, never by the presence or
        # absence of a resumed event. A refusal is a governance outcome, not
        # an error: the span stays unset/ok.
        outcome = graph.invoke(Command(resume=decision), config)
        print("    ->", outcome["result"])


def run_expired_flow(checkpointer: InMemorySaver):
    """Pause with a deadline in the interrupt payload; nobody ever answers.

    The expiry sweep reads the pending interrupt and its deadline back from
    the thread's persisted state. The deadline exists in that state, so the
    producer may emit `expired`; it is a record of absence, not an inference
    from absence of record.
    """
    print("  [lifecycle] expired: interrupt with deadline -> deadline passes -> expired")
    config = {"configurable": {"thread_id": "exec-expired-01"}}
    thread_id = config["configurable"]["thread_id"]

    with _invoke_agent_span() as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", AGENT_NAME)
        span.set_attribute("gen_ai.agent.execution.id", thread_id)
        graph = build_graph(checkpointer)
        pause_id, _checkpoint_id = _pause_until_interrupt(graph, config, "rotate signing keys")

    # Later, an expiry sweep inspects the thread's persisted state: the pause
    # is still pending and its payload carries the configured deadline.
    state = build_graph(checkpointer).get_state(config)
    (pending,) = state.interrupts
    deadline = datetime.fromisoformat(pending.value["deadline"])
    now = deadline + timedelta(minutes=1)  # the sweep runs after the deadline
    if now > deadline:
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": thread_id,
                "gen_ai.agent.pause.id": pending.id,
                "gen_ai.agent.pause.resolution": "expired",
                "gen_ai.agent.name": AGENT_NAME,
            },
        )
        assert pending.id == pause_id
        print("    -> expired; execution terminates at the boundary")


def main():
    tp, lp, mp = setup_otel()
    print("agent-lifecycle reference scenario (LangGraph)")
    try:
        run_approved_flow(InMemorySaver())
        run_refused_flow(InMemorySaver())
        run_expired_flow(InMemorySaver())
    finally:
        flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
