from agent_lab.tools import execute_tool
from agent_lab.agent import AgentState, Observation
from agent_lab.memory import Memory
from agent_lab.models import Action, Observation
from agent_lab.trajectory import Trajectory


MAX_RETRIES = 1

def investigate_compound(
    compound: str,
    state: AgentState,
    memory: Memory,
    trajectory: Trajectory,
) -> Observation:
    
    if compound in state.tested_compounds:
        return Observation(success=True, result=state.tested_compounds[compound])

    action = Action(tool="run_assay", arguments={"compound": compound})

    observation = execute_tool(action)

    trajectory.record(action, observation)

    if observation.success:
        state.tested_compounds[compound] = observation.result

        memory.add(
            kind="assay_result",
            content=f"{compound}: {observation.result}",
        )
    else:
        state.retry_counts[compound] = state.retry_counts.get(compound, 0) + 1
        return investigate_compound(compound, state, memory, trajectory)

    return observation

    