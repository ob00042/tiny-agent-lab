from agent_lab.state import AgentState


def should_escalate(
    state: AgentState,
) -> bool:
    return any(
        retry_count >= 2
        for retry_count
        in state.retry_counts.values()
    )