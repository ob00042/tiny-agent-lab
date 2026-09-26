from agent_lab.agent import run_agent

def test_agent_finds_successful_compound():
    run = run_agent()

    assert run.state.finished == True
    assert run.state.selected_compound == "B"
    assert run.state.tested_compounds == {
        "A": 0.42,
        "B": 0.81,
    }
    assert len(run.memory.items) == 2
    assert len(run.trajectory.steps) == 2
