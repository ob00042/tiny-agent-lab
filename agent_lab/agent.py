from agent_lab.policy import choose_next_action, choose_next_compound
from agent_lab.state import AgentState
from agent_lab.models import Action, Observation
from agent_lab.tools import execute_tool

from dataclasses import dataclass

from agent_lab.memory import Memory
from agent_lab.skills import investigate_compound
from agent_lab.trajectory import Trajectory

from agent_lab.proposer_critic import Proposer, Critic, choose_approved_proposal

from agent_lab.reflection import reflect
from agent_lab.planner import Planner, should_replan

from agent_lab.control import should_escalate


def apply_observation(state: AgentState, action: Action, observation: Observation) -> None:
    if action.tool == "run_assay":
        compound = action.arguments["compound"]
        if not observation.success:
            return

        score = observation.result

        state.tested_compounds[compound] = score
        state.current_compound = compound

        if score >= 0.75:
            state.selected_compound = compound
            state.finished = True
    else:
        return


# def run_agent() -> AgentState:
#     state = AgentState()

#     while not state.finished:
#         action = choose_next_action(state)

#         if action is None:
#             break

#         observation = execute_tool(action)
#         apply_observation(state=state, 
#                         action=action, 
#                         observation=observation)

#     return state


@dataclass
class AgentRun:
    state: AgentState
    memory: Memory
    trajectory: Trajectory


def run_agent() -> AgentRun:
    '''
    1. initialize state, memory, trajectory

    2. ask policy for next compound

    3. if policy returns None:
        stop

    4. investigate compound using the skill

    5. inspect the observation

    6. if observation is scientifically good enough:
        mark goal complete

    7. otherwise ask policy again
    '''
    state = AgentState()
    memory = Memory()
    trajectory = Trajectory()
    agent_run = AgentRun(state=state, memory=memory, trajectory=trajectory)
    proposer = Proposer()
    critic = Critic()

    planner = Planner()
    plan = planner.create_plan() # fake plan, doesn't do anything

    while state.finished == False:

        # compound = choose_next_compound(state=state)
        proposal = choose_approved_proposal(proposer=proposer, critic=critic, state=state)
        if proposal is None:
            return agent_run
        compound = proposal.action.arguments["compound"]

        if compound is None:
            return agent_run

        observation = investigate_compound(compound=compound, state=state, memory=memory, trajectory=trajectory)

        if observation.success and observation.result > 0.75:
            state.finished = True
            state.selected_compound = compound

        reflect(trajectory=trajectory, memory=memory)

        if should_replan(observation):
            plan = planner.create_plan() # fake plan, doesn't do anything

        if should_escalate(state):
            state.requires_human = True
            break

    return agent_run
    