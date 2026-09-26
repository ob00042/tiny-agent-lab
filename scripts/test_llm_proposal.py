# uv run python -m scripts.test_llm_proposal
from agent_lab.llm import LocalLLMClient
from agent_lab.proposer_critic import LLMProposer
from agent_lab.state import AgentState


client = LocalLLMClient()

state = AgentState()

llm_proposer = LLMProposer(client=client)
proposal = llm_proposer.propose(state)

print(proposal)
