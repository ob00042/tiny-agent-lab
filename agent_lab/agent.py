from agent_lab.policy import choose_next_action
from agent_lab.state import AgentState
from agent_lab.models import Action, Observation
from agent_lab.tools import execute_tool


def apply_observation(state: AgentState, action: Action, observation: Observation) -> None:
    if action.tool == "run_assay":
        compound = action.arguments["compound"]
        if not observation.success:
            return

        score = observation.result

        state.tested_compounds[compound] = score
        state.current_compound = compound

        if score >= 0.75:
            state.selected_compound = compound
            state.finished = True
    else:
        return


def run_agent() -> AgentState:
    state = AgentState()

    while not state.finished:
        action = choose_next_action(state)

        if action is None:
            break

        observation = execute_tool(action)
        apply_observation(state=state, 
                        action=action, 
                        observation=observation)

    return state
