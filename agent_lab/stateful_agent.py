from agent_lab.simple_loop import run_assay
from agent_lab.state import AgentState


def run_agent() -> AgentState:
    state = AgentState()

    for compound in ["A", "B", "C"]:
        state.current_compound = compound
        score = run_assay(compound)
        print(f"Observed compound {compound}: {score}")
        state.tested_compounds[compound] = score
        if score > 0.75:
            state.selected_compound = compound
            state.finished = True
            break
    
    return state

# uv run python -c "from agent_lab.stateful_agent import run_agent; print(run_agent())"
