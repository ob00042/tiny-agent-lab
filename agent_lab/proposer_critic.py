from dataclasses import dataclass

from agent_lab.models import Action
from agent_lab.state import AgentState
from agent_lab.domain import CANDIDATES, PREDICTED_SCORES, TARGET_SCORE
from agent_lab.llm import LocalLLMClient
from agent_lab.decision_log import DecisionLog
from agent_lab.decision_models import Proposal, Critique


class Proposer:
    def propose(
        self,
        state: AgentState,
        feedback: str | None = None,
    ) -> Proposal:
        '''
        look at candidates
        choose first untested compound
        return proposal
        '''
        for candidate in CANDIDATES:
            if candidate not in state.tested_compounds:
                return Proposal(
                    action=Action(tool="run_assay", arguments={"compound": candidate}), 
                    reason=f"Compound {candidate} has not been tested yet."
                )

        return None

    
class Critic:
    def evaluate(
        self,
        state: AgentState,
        proposal: Proposal,
    ) -> Critique:
        '''
        1. Is tool == "run_assay"?
        2. Is compound valid?
        3. Has it already been tested?
        '''
        if proposal.action.tool != "run_assay":
            return Critique(
                accepted=False,
                reason="Action is not a run_assay action."
            )
        
        if proposal.action.arguments.get("compound") not in CANDIDATES:
            return Critique(
                accepted=False,
                reason="Compound is not a valid candidate."
            )
        
        if proposal.action.arguments.get("compound") in state.tested_compounds:
            return Critique(
                accepted=False,
                reason="Compound has already been tested."
            )
        
        return Critique(
            accepted=True,
            reason="Proposal is valid."
        )


def choose_approved_proposal(
    proposer: Proposer,
    critic: Critic,
    state: AgentState,
    decision_log: DecisionLog,
    max_attempts: int = 3,
) -> Proposal:
    
    feedback = None

    for _ in range(max_attempts):
        proposal = proposer.propose(state, feedback=feedback)
        critique = critic.evaluate(state=state, proposal=proposal)

        decision_log.record(proposal=proposal, critique=critique)

        if critique.accepted:
            return proposal
        
        feedback = critique.reason
    
    # raise RuntimeError("No proposal was accepted")


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "compound": {
            "type": "string",
            "enum": ["A", "B", "C"],
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "compound",
        "reason",
    ],
    "additionalProperties": False,
}


PROPOSER_SYSTEM_PROMPTS = {
    "v1": """
    You are the proposal component of a scientific agent.

    Your job is to choose exactly one compound to assay next.

    Rules:
    - Only choose A, B, or C.
    - Prefer candidates whose predicted score suggests they may reach the target.
    - Never treat predicted scores as experimental results.
    - Do not choose a compound that has already been tested.
    - Return only the requested structured output.
    """
}


class LLMProposer:
    def __init__(
        self,
        client: LocalLLMClient,
        prompt_version: str,
    ) -> None:
        self.client = client
        self.prompt_version = prompt_version

    def _build_state_prompt(
        self,
        state: AgentState,
        feedback: str | None = None,
    ) -> str:
        predictions = "\n".join(
            f"{compound}: {PREDICTED_SCORES[compound]}"
            for compound in CANDIDATES
        )

        if state.tested_compounds:
            tested = "\n".join(
                f"{compound}: {score}"
                for compound, score
                in state.tested_compounds.items()
            )
        else:
            tested = "None"

        if feedback:
            feedback_section = f"""
            The previous proposal was rejected.

            Critic feedback:
            {feedback}

            Choose a different valid action that addresses this feedback.
            """
        else:
            feedback_section = ""

        return f"""
                Target assay score: {TARGET_SCORE}

                Predicted scores:
                {predictions}

                Already tested compounds and observed results:
                {tested}

                {feedback_section}

                Choose the next compound to assay.
                """

    @property
    def system_prompt(self) -> str:
        return PROPOSER_SYSTEM_PROMPTS[self.prompt_version]
    
    def propose(
        self,
        state: AgentState,
        feedback: str | None = None,
    ) -> Proposal:
        response = self.client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self._build_state_prompt(state=state, feedback=feedback),
                },
            ],
            schema=PROPOSAL_SCHEMA,
        )

        return Proposal(
            action=Action(
                tool="run_assay",
                arguments={
                    "compound": response["compound"],
                },
            ),
            reason=response["reason"],
        )
