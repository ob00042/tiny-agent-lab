from agent_lab.decision_log import DecisionLog
from agent_lab.decision_models import Proposal, Critique
from agent_lab.models import Action


def test_decision_log_records_proposal_and_critique():
    decision_log = DecisionLog()

    proposal = Proposal(
        action=Action(
            tool="run_assay",
            arguments={"compound": "B"},
        ),
        reason="B looks promising",
    )

    critique = Critique(
        accepted=True,
        reason="Valid untested compound",
    )

    decision_log.record(proposal, critique)

    assert len(decision_log.records) == 1
    assert decision_log.records[0].index == 0
    assert decision_log.records[0].proposal == proposal
    assert decision_log.records[0].critique == critique


def test_decision_log_indexes_multiple_records():
    decision_log = DecisionLog()

    proposal = Proposal(
        action=Action(
            tool="run_assay",
            arguments={"compound": "B"},
        ),
        reason="B looks promising",
    )

    critique = Critique(
        accepted=True,
        reason="Valid untested compound",
    )

    decision_log.record(proposal, critique)
    decision_log.record(proposal, critique)

    assert len(decision_log.records) == 2
