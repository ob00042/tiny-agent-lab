from agent_lab.models import Action, Observation

COMPOUNDS = {
    "A": 0.42,
    "B": 0.81,
    "C": 0.63,
}

CALL_COUNTS: dict[str, int] = {}


def run_assay(compound: str) -> float:
    '''
    1. look up the compound;
    2. return the score;
    3. raise an error if it doesn't exist.
    '''
    if compound not in COMPOUNDS:
        raise ValueError(f"Unknown compound: {compound}")

    CALL_COUNTS[compound] = CALL_COUNTS.get(compound, 0) + 1
    if compound == "C" and CALL_COUNTS[compound] == 1:
        raise RuntimeError("Instrument timeout")

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

