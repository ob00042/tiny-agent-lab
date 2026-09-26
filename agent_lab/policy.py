from agent_lab.models import Action
from agent_lab.state import AgentState

CANDIDATES = ["A", "B", "C"]


def choose_next_compound(
    state: AgentState,
) -> str | None:
    for compound in CANDIDATES:
        if compound not in state.tested_compounds:
            return compound

    return None


def choose_next_action(state: AgentState) -> Action | None:
    '''
    Deterministic (for now)
    if A untested → test A
    else if B untested → test B
    else if C untested → test C
    else → no action
    '''
    for compound in ["A", "B", "C"]:
        if compound not in state.tested_compounds:
            return Action(tool="run_assay", arguments={"compound": compound})
    return None
