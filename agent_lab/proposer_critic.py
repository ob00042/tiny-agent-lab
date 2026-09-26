from dataclasses import dataclass

from agent_lab.models import Action
from agent_lab.state import AgentState


CANDIDATES = ["A", "B", "C"]


@dataclass
class Proposal:
    action: Action
    reason: str


@dataclass
class Critique:
    accepted: bool
    reason: str


class Proposer:
    def propose(
        self,
        state: AgentState,
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
    max_attempts: int = 3,
) -> Proposal:
    '''
    propose
    ↓
    critic
    ├── accept → return proposal
    └── reject → ask proposer again
    '''
    for _ in range(max_attempts):
        proposal = proposer.propose(state)
        critique = critic.evaluate(state=state, proposal=proposal)
        if critique.accepted:
            return proposal
        continue
