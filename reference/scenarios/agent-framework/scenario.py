"""Native telemetry scenario for Microsoft Agent Framework."""

import asyncio
import os
from typing import Annotated

from reference_shared import flush_and_shutdown, reference_event_logger, setup_otel

_LIFECYCLE_LOGGER = "agent_framework.lifecycle.reference"

MOCK_BASE_URL = os.environ["MOCK_LLM_URL"] + "/v1"


async def run_agent_tool_call():
    """Scenario: Agent Framework agent execution with native telemetry."""
    from agent_framework import Agent, tool
    from agent_framework.observability import enable_sensitive_telemetry
    from agent_framework.openai import OpenAIChatClient

    print("  [agent_run] agent with tool calling (native telemetry)")

    enable_sensitive_telemetry(force=True)

    @tool(approval_mode="never_require")
    def get_weather(
        location: Annotated[str, "The location to get the weather for."],
    ) -> str:
        """Get the weather for a given location."""
        return f"Sunny in {location}"

    client = OpenAIChatClient(
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )
    agent = Agent(
        client=client,
        id="weather-agent",
        name="WeatherAgent",
        description="Answers weather questions with a function tool.",
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    result = await agent.run(
        "What's the weather in Seattle?",
        options={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 64,
        },
    )
    print(f"    -> {result.text[:60]}")


async def run_tool_call():
    """Scenario: Agent Framework chat client tool calling with native telemetry."""
    from agent_framework import Message, tool
    from agent_framework.observability import enable_sensitive_telemetry
    from agent_framework.openai import OpenAIChatCompletionClient

    print("  [chat_tool_call] chat client with tool calling (native telemetry)")

    enable_sensitive_telemetry(force=True)

    @tool(approval_mode="never_require")
    def get_weather(
        location: Annotated[str, "The location to get the weather for."],
    ) -> str:
        """Get the weather for a given location."""
        return f"Sunny in {location}"

    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )
    response = await client.get_response(
        [Message(role="user", contents=["What's the weather in Seattle?"])],
        options={
            "tools": [get_weather],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 64,
            "seed": 7,
            "stop": ["<END>"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        },
    )
    print(f"    -> {response.text[:60]}")


async def run_chat_completion_agent_tool_call():
    """Scenario: Agent Framework agent execution through Chat Completions."""
    from agent_framework import Agent, tool
    from agent_framework.observability import enable_sensitive_telemetry
    from agent_framework.openai import OpenAIChatCompletionClient

    print("  [agent_chat_completion] agent with Chat Completions (native telemetry)")

    enable_sensitive_telemetry(force=True)

    @tool(approval_mode="never_require")
    def get_weather(
        location: Annotated[str, "The location to get the weather for."],
    ) -> str:
        """Get the weather for a given location."""
        return f"Sunny in {location}"

    client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )
    agent = Agent(
        client=client,
        id="weather-agent-chat-completions",
        name="WeatherAgentChatCompletions",
        description="Answers weather questions with a function tool.",
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    result = await agent.run(
        "What's the weather in Seattle?",
        options={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 64,
            "seed": 7,
            "stop": ["<END>"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        },
    )
    print(f"    -> {result.text[:60]}")


async def run_agent_workflow():
    """Scenario: Agent Framework workflow execution with native telemetry."""
    from agent_framework import Agent, WorkflowBuilder
    from agent_framework.observability import enable_sensitive_telemetry
    from agent_framework.openai import OpenAIChatClient

    print("  [workflow] two-agent workflow (native telemetry)")

    enable_sensitive_telemetry(force=True)

    client = OpenAIChatClient(
        model="gpt-4o-mini",
        base_url=MOCK_BASE_URL,
        api_key="mock-key",
    )
    writer_agent = Agent(
        client=client,
        name="writer",
        instructions="You are a concise copy writer.",
    )
    reviewer_agent = Agent(
        client=client,
        name="reviewer",
        instructions="You review slogans and suggest one short improvement.",
    )
    workflow = (
        WorkflowBuilder(
            start_executor=writer_agent,
            name="slogan_review_workflow",
            description="Drafts and reviews a short slogan.",
            output_from=[reviewer_agent],
        )
        .add_edge(writer_agent, reviewer_agent)
        .build()
    )

    result = await workflow.run("Create a slogan for a compact electric van.")
    outputs = result.get_outputs()
    if outputs:
        print(f"    -> {str(outputs[0])[:60]}")


async def run_tool_approval_reference():
    """Scenario: Agent Framework approval gate, pause, resolution, and resume."""
    from agent_framework import Agent, Message, tool
    from agent_framework.openai import OpenAIChatClient

    print("  [approval_gate] tool approval gate (pause, resolution, resume)")

    @tool(approval_mode="always_require")
    def refund_order(
        order_id: Annotated[str, "The order to refund."],
    ) -> str:
        """Refund an order."""
        return f"refunded {order_id}"

    client = OpenAIChatClient(model="gpt-4o-mini", base_url=MOCK_BASE_URL, api_key="mock-key")
    agent = Agent(
        client=client,
        id="refund-agent",
        name="RefundAgent",
        description="Refunds orders behind an approval gate.",
        instructions="You refund orders.",
        tools=[refund_order],
    )

    # The framework generates the session id and keeps it across the suspend
    # and the later resume, so it identifies the run rather than one response.
    session = agent.create_session()

    # A tool declared always_require suspends the run: the framework returns
    # its own approval request instead of invoking the tool.
    suspended = await agent.run("Refund order 4417.", session=session)
    (approval_request,) = suspended.user_input_requests

    # The framework labels the request it raised; an approval request means the
    # run waits on a person, any other user input request on another system.
    pause_reason = "human_input" if approval_request.type == "function_approval_request" else "external_system"

    reference_event_logger(_LIFECYCLE_LOGGER).emit(
        event_name="gen_ai.agent.paused",
        body="paused",
        attributes={
            "gen_ai.agent.execution.id": session.session_id,
            "gen_ai.agent.pause.id": approval_request.id,
            "gen_ai.agent.pause.reason": pause_reason,
            "gen_ai.agent.id": agent.id,
            "gen_ai.agent.name": agent.name,
        },
    )

    # The decision travels back in the framework's own response content, whose
    # approved field carries the outcome.
    decision = approval_request.to_function_approval_response(approved=True)
    await agent.run(Message(role="user", contents=[decision]), session=session)

    reference_event_logger(_LIFECYCLE_LOGGER).emit(
        event_name="gen_ai.agent.pause.resolved",
        body="pause.resolved",
        attributes={
            "gen_ai.agent.execution.id": session.session_id,
            "gen_ai.agent.pause.id": approval_request.id,
            "gen_ai.agent.pause.resolution": "approved" if decision.approved else "refused",
            "gen_ai.agent.id": agent.id,
            "gen_ai.agent.name": agent.name,
        },
    )
    reference_event_logger(_LIFECYCLE_LOGGER).emit(
        event_name="gen_ai.agent.resumed",
        body="resumed",
        attributes={
            "gen_ai.agent.execution.id": session.session_id,
            "gen_ai.agent.resumed_from.type": "pause",
            "gen_ai.agent.resumed_from.id": approval_request.id,
            "gen_ai.agent.id": agent.id,
            "gen_ai.agent.name": agent.name,
        },
    )
    print(f"    -> resolved approval {approval_request.id}")


def main():
    print("=== Native Telemetry: Microsoft Agent Framework ===")

    tp, lp, mp = setup_otel()

    asyncio.run(run_agent_tool_call())
    asyncio.run(run_tool_approval_reference())
    asyncio.run(run_tool_call())
    asyncio.run(run_chat_completion_agent_tool_call())
    asyncio.run(run_agent_workflow())

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
