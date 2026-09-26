'''
accepted immediately
rejected once then accepted
all attempts rejected
'''
from agent_lab.proposer_critic import Proposer, Critic, choose_approved_proposal, Proposal, Critique
from agent_lab.state import AgentState
from agent_lab.models import Action
from agent_lab.decision_log import DecisionLog


class FakeProposer:
    def __init__(self):
        self.calls = 0

    def propose(self, state, feedback=None):
        self.calls += 1

        return Proposal(
            action=Action(
                tool="run_assay",
                arguments={"compound": "A"},
            ),
            reason="test",
        )


class AlwaysAcceptCritic:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, proposal):
        self.calls += 1

        return Critique(
            accepted=True,
            reason="accepted",
        )


def test_choose_approved_proposal_accept_immediately():
    proposer = FakeProposer()
    critic = AlwaysAcceptCritic()
    state = AgentState()
    decision_log = DecisionLog()

    proposal = choose_approved_proposal(proposer=proposer, 
                                        critic=critic, 
                                        state=state,
                                        decision_log=decision_log)

    assert proposal is not None
    assert proposal.action.arguments["compound"] == "A"
    assert proposer.calls == 1
    assert critic.calls == 1


class RejectOnceCritic:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, proposal):
        self.calls += 1

        if self.calls == 1:
            return Critique(
                accepted=False,
                reason="failed"
            )

        return Critique(
            accepted=True,
            reason="accepted",
        )


def test_choose_approved_proposal_reject_once_then_accept():
    proposer = FakeProposer()
    critic = RejectOnceCritic()
    state = AgentState()
    decision_log = DecisionLog()

    proposal = choose_approved_proposal(proposer=proposer, 
                                        critic=critic, 
                                        state=state,
                                        decision_log=decision_log)

    assert proposal is not None
    assert proposal.action.arguments["compound"] == "A"
    assert proposer.calls == 2
    assert critic.calls == 2


class RejectAllCritic():
    def __init__(self):
        self.calls = 0

    def evaluate(self, state, proposal):
        self.calls += 1

        return Critique(
            accepted=False,
            reason="failed"
        )


def test_choose_approved_proposal_reject_all():
    proposer = FakeProposer()
    critic = RejectAllCritic()
    state = AgentState()
    decision_log = DecisionLog()

    proposal = choose_approved_proposal(proposer=proposer, 
                                        critic=critic, 
                                        state=state,
                                        decision_log=decision_log)

    assert proposal is None
    assert proposer.calls == 3
    assert critic.calls == 3


class FeedbackRecordingProposer:
    def __init__(self):
        self.feedback_received = []

    def propose(self, state, feedback=None):
        self.feedback_received.append(feedback)

        return Proposal(
            action=Action(
                tool="run_assay",
                arguments={"compound": "A"},
            ),
            reason="test",
        )


def test_choose_approved_proposal_proposer_takes_feedback():
    proposer = FeedbackRecordingProposer()
    critic = RejectAllCritic()
    state = AgentState()
    decision_log = DecisionLog()

    proposal = choose_approved_proposal(proposer=proposer, 
                                        critic=critic, 
                                        state=state, 
                                        decision_log=decision_log)

    assert proposal is None
    assert len(proposer.feedback_received) == 3
    assert proposer.feedback_received == [None, "failed", "failed"]
