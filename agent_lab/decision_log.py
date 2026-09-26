from dataclasses import dataclass, field

from agent_lab.decision_models import (
    Critique,
    Proposal,
)


'''
Decision log

"What did the decision system propose and why?"


Trajectory

"What actions actually executed and what happened?"
'''


@dataclass
class DecisionRecord:
    index: int
    proposal: Proposal
    critique: Critique


@dataclass
class DecisionLog:
    records: list[DecisionRecord] = field(
        default_factory=list
    )

    def record(
        self,
        proposal: Proposal,
        critique: Critique,
    ) -> None:
        self.records.append(
            DecisionRecord(
                index=len(self.records),
                proposal=proposal,
                critique=critique,
            )
        )
