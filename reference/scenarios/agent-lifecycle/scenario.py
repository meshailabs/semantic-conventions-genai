"""Reference implementation for agent lifecycle events (#159).

Models a durable approval gate in front of a long-running agent: execution
pauses awaiting a human decision, state is checkpointed, and the run either
resumes in a new execution segment (approved), or terminates at the boundary
(refused / expired). No LLM calls are involved; the lifecycle surface is
framework-neutral by design.
"""

from opentelemetry.trace import SpanKind
from reference_shared import flush_and_shutdown, reference_event_logger, reference_tracer, setup_otel

_reference_tracer = reference_tracer()
_LIFECYCLE_LOGGER = "gen_ai.agent.lifecycle.reference"


def _emit(event_name: str, attributes: dict) -> None:
    reference_event_logger(_LIFECYCLE_LOGGER).emit(
        event_name=event_name,
        body=event_name.rsplit(".", 1)[-1],
        attributes=attributes,
    )


def _invoke_agent_span(name: str):
    return _reference_tracer.start_as_current_span(name, kind=SpanKind.INTERNAL)


def run_approved_flow():
    """Pause for approval, checkpoint, approve, resume in a NEW segment.

    The resumed segment carries the same execution.id and identifies the
    suspended boundary via the flat resumed_from pair, reconstructing the
    suspend/resume chain without resolving the original (closed) span.
    """
    print("  [lifecycle] approved: pause -> checkpoint -> approve -> resume in new segment")
    execution_id = "exec_ref_approved_01"
    pause_id = "pause_ref_01"
    checkpoint_id = "ck_ref_01"

    # Segment 1: runs until the approval gate suspends it.
    with _invoke_agent_span("invoke_agent payment_reconciler") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "payment_reconciler")
        span.set_attribute("gen_ai.agent.execution.id", execution_id)
        _emit(
            "gen_ai.agent.paused",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.reason": "human_input",
                "gen_ai.agent.pause.id": pause_id,
            },
        )
        _emit(
            "gen_ai.agent.checkpointed",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.checkpoint.id": checkpoint_id,
            },
        )
    # Segment 1's span is now closed: the process hosting it may exit here.

    # The decision arrives out of band (approval UI, ticket, chat).
    # Segment 2: a different worker picks the run up from the checkpoint.
    with _invoke_agent_span("invoke_agent payment_reconciler") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "payment_reconciler")
        span.set_attribute("gen_ai.agent.execution.id", execution_id)
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.id": pause_id,
                "gen_ai.agent.pause.resolution": "approved",
            },
        )
        # The resumed segment names the PAUSE it continues from, keeping the
        # resolution -> resume linkage unambiguous when one execution holds
        # several pauses or checkpoints. The checkpointed event above stands as
        # an independent persistence fact; `resumed_from.type=checkpoint` is
        # for durable recovery with no decision boundary (e.g. crash restart).
        _emit(
            "gen_ai.agent.resumed",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.resumed_from.type": "pause",
                "gen_ai.agent.resumed_from.id": pause_id,
            },
        )
        print("    -> resumed from", pause_id)


def run_refused_flow():
    """Pause for approval; the human says no. Execution ends at the boundary.

    A refusal is a governance outcome, not an error: the span ends
    unset/ok and the terminal fact is carried by the resolution event.
    """
    print("  [lifecycle] refused: pause -> refuse -> no resumed segment")
    execution_id = "exec_ref_refused_01"
    pause_id = "pause_ref_02"

    with _invoke_agent_span("invoke_agent db_migrator") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "db_migrator")
        span.set_attribute("gen_ai.agent.execution.id", execution_id)
        _emit(
            "gen_ai.agent.paused",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.reason": "human_input",
                "gen_ai.agent.pause.id": pause_id,
            },
        )
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.id": pause_id,
                "gen_ai.agent.pause.resolution": "refused",
            },
        )
        print("    -> refused; execution terminates at the boundary")


def run_expired_flow():
    """Pause with a configured deadline; nobody answers before it passes.

    The producer emits `expired` because a deadline existed and passed: a
    record of absence, not an inference from absence of record.
    """
    print("  [lifecycle] expired: pause with deadline -> deadline passes -> expired")
    execution_id = "exec_ref_expired_01"
    pause_id = "pause_ref_03"

    with _invoke_agent_span("invoke_agent contract_summarizer") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.agent.name", "contract_summarizer")
        span.set_attribute("gen_ai.agent.execution.id", execution_id)
        _emit(
            "gen_ai.agent.paused",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.reason": "human_input",
                "gen_ai.agent.pause.id": pause_id,
            },
        )
        # The gate's configured deadline passes with no decision.
        _emit(
            "gen_ai.agent.pause.resolved",
            {
                "gen_ai.agent.execution.id": execution_id,
                "gen_ai.agent.pause.id": pause_id,
                "gen_ai.agent.pause.resolution": "expired",
            },
        )
        print("    -> expired; execution terminates at the boundary")


def main():
    tp, lp, mp = setup_otel()
    print("agent-lifecycle reference scenario")
    try:
        run_approved_flow()
        run_refused_flow()
        run_expired_flow()
    finally:
        flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
