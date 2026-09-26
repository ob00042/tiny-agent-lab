from agent_lab.proposer_critic import LLMProposer
from agent_lab.state import AgentState
from agent_lab.config import AgentConfig


class FakeLLMClient:
    def chat_json(
        self,
        messages,
        schema,
    ):
        return {
            "compound": "B",
            "reason": "B has the highest predicted score.",
        }


def test_llm_proposer_builds_proposal():
    client = FakeLLMClient()
    state = AgentState()
    config = AgentConfig()

    llm_proposer = LLMProposer(client=client, prompt_version=config.proposer_prompt_version)

    proposal = llm_proposer.propose(state=state)

    assert proposal.action.tool == "run_assay"
    assert proposal.action.arguments["compound"] == "B" 
