# environment
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


def run_agent() -> str | None:
    '''
    for candidate
        run assay
        inspect observation
        if good enough (> 0.75)
            terminate
    '''
    for compound in ["A", "B", "C"]:
        score = run_assay(compound)

        print(f"Observed compound {compound}: {score}")

        if score > 0.75:
            return compound

    return None
