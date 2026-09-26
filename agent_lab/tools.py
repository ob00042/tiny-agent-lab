from agent_lab.models import Action, Observation

COMPOUNDS = {
    "A": 0.42,
    "B": 0.81,
    "C": 0.63,
}


def run_assay(compound: str) -> float:
    '''
    1. look up the compound;
    2. return the score;
    3. raise an error if it doesn't exist.
    '''
    if compound not in COMPOUNDS:
        raise ValueError(f"Unknown compound: {compound}")

    return COMPOUNDS[compound]


TOOLS = {
    "run_assay": run_assay,
}


def execute_tool(action: Action) -> Observation:
    tool = TOOLS.get(action.tool)

    if tool is None:
        return Observation(
            success=False,
            error=f"Unknown tool: {action.tool}",
        )

    try:
        result = tool(**action.arguments)
        return Observation(
            success=True,
            result=result,
        )
    except Exception as exc:
        return Observation(
            success=False,
            error=str(exc),
        )

