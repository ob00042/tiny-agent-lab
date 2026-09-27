# uv run python -m scripts.run_local_agent
from agent_lab.agent import run_agent
from agent_lab.llm import LocalLLMClient
from agent_lab.proposer_critic import (
    Critic,
    LLMProposer,
)
from agent_lab.config import AgentConfig

config = AgentConfig()
client = LocalLLMClient(temperature=config.temperature)
critic = Critic()
proposer = LLMProposer(client=client, prompt_version=config.proposer_prompt_version)

run = run_agent(proposer=proposer, critic=critic, config=config, client=client)

print("Finished:", run.state.finished)
print(
    "Selected:",
    run.state.selected_compound,
)
print(
    "Tested:",
    run.state.tested_compounds,
)
print(
    "Trajectory steps:",
    len(run.trajectory.steps),
)
print(
    "Memory:",
    run.memory
)
