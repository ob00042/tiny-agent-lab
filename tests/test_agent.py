from agent_lab.agent import run_agent
from agent_lab.proposer_critic import Proposer, Critic
from agent_lab.config import AgentConfig
from agent_lab.llm import LocalLLMClient

def test_agent_finds_successful_compound():
    config = AgentConfig()
    proposer = Proposer()
    critic = Critic()
    client = LocalLLMClient(temperature=config.temperature)
    run = run_agent(proposer=proposer, critic=critic, config=config, client=client)

    assert run.state.finished == True
    assert run.state.selected_compound == "B"
    assert run.state.tested_compounds == {
        "A": 0.42,
        "B": 0.81,
    }
    assert len(run.memory.items) == 2
    assert len(run.trajectory.steps) == 2
