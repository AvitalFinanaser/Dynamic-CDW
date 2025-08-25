import json
from pathlib import Path
from typing import List, Tuple

from agents import *
from events import *
from paragraphs import Paragraph, ParagraphWithScore
from rules import *
from simulation.scheduler import Scheduler


def test_event():
    ## How to define a new event e:= <agent, paragraph, vote>
    agent = Agent(1)
    paragraph = Paragraph(paragraph_id=1, text="This is a sample paragraph", name="p1")
    event = Event(agent, paragraph, +1)
    print(event)
    # getters
    print(f"Agent of event {event} is:\n{event.agent}")
    print(f"Paragraph of event {event} is:\n{event.paragraph}")
    print(f"Vote of event {event} is:\n{event.vote}")
    # setters
    event.agent = AgentInterval(agent_id=2)
    print(f"Event's agent was updated:\n{event}")
    event.paragraph = ParagraphWithScore(text="Hi", paragraph_id=5,name="p5")
    print(f"Event's paragraph was updated:\n{event}")
    event.vote = -1
    print(f"Event's agent was updated:\n{event}")



def test_event_list():
    ## Manually define an event list E:
    event_list = EventList()
    print(event_list)
    # add event and print list
    agent = Agent(1)
    paragraph = Paragraph(paragraph_id=1, text="This is a sample paragraph", name="p1")
    event = Event(agent, paragraph, +1)
    event_list.add_event(event)
    print(event_list)
    event_list.add_event(Event(Agent(2), Paragraph(paragraph_id=2, text="p2", name="p2"), -1))
    print(f"\nUpdated event list:\n{event_list}")
    # p+
    print(f"\nP^+:\n{event_list.p_plus(paragraph)}")
    # p-
    print(f"\nP^-:\n{event_list.p_minus(paragraph)}")
    # p0
    print(f"\nP^0:\n{event_list.p_avoid(paragraph)}")
    # P()
    print(f"\nP(E):\n{event_list.P()}")
    # E_p
    event_list.add_event(Event(Agent(3), paragraph, -1))
    print(f"\nE_p:\n{event_list.E_p(paragraph)}")
    # E_a
    print(f"\nE_a:\n{event_list.E_a(agent)}")
    # E_i
    print(f"\nE_i:\n{event_list.E_i(1)}")
    # A_E
    print(f"\nA_E:\n{event_list.A_E()}")
    # A_p
    print(f"\nA_p:\n{event_list.A_p(paragraph)}")
    # events_df
    event_list.add_event(Event(agent,paragraph, -1))
    print(f"\ndf_events:\n{event_list.events_df()}")
    # get vote - last stance
    print(f"\nLast vote of agent {agent.name} on paragraph {paragraph.name} is:\n{event_list.get_vote(a=agent, p=paragraph)}")
    # get number of events
    print(f"\nNumber of events:\n{len(event_list.events)}")
    # Summary of E
    agents, paragraphs, summary_matrix = event_list.sum()
    print(f"\nSum(E):\n{summary_matrix}")
    # nice print for summary
    print(f"\n{event_list.print_sum_matrix()}")
    # tally of E
    print(f"\nTally matrix:\n{event_list.tally()}")
    # nice print for tally
    print(f"\n{event_list.print_tally_matrix()}")
    # Active agents
    print(f"Active agents:\n{event_list.activeAgents()}")
    # Na of an agent
    event_list.add_event(Event(agent, Paragraph(paragraph_id=3, text="p3", name="p3"), 1))
    print(f"Na for the agent:\n{event_list.agentNa(agent=agent)} \nevent list:\n {event_list}")
    # Relative tally matrix
    print(f"\nRelative Tally matrix:\n{event_list.tally_r()}")
    # System state
    print(f"\nGlobal system state:\n{event_list.get_global_state()}")
    # System state for an agent
    print(f"\nAgent view of system state:\n{event_list.get_current_state_for_prompt(agent)}")
    # Finding paragraph by id
    retrieved_p = event_list.p_by_id(1)
    print(f"\nRetrieved Paragraph by ID 1:\n{retrieved_p}, type {type(retrieved_p)})")
    # Event_from_action
    mock_action = {
        "action_id": 1,
        "type": "VOTE",
        "paragraph_id": 1,
        "content": "This is a sample paragraph",
        "vote": "UPVOTE",
        "reasoning": "Costs are too high at this stage."
    }
    generated_event = event_list.event_from_action(agent, mock_action)
    print(f"\nGenerated Event:{generated_event}, type {type(generated_event)}")


def test_loading_eventlist() -> Tuple[List[LLMAgent], EventList]:
    # Define root path
    save_root = Path("datasets/event_lists") / "event_list_1"

    # Load Agents
    with open(save_root / "agents.json", "r") as f:
        agents_data = json.load(f)

    community = []
    for data in agents_data:
        agent = LLMAgent(
            agent_id=data["agent_id"],
            profile=data["profile"],
            topic=data["topic"],
            topic_position=data["topic_position"]
        )
        community.append(agent)

    # Load Paragraphs
    with open(save_root / "paragraphs.json", "r") as f:
        paragraphs_data = json.load(f)

    paragraphs = {}
    for data in paragraphs_data:
        p = Paragraph(text=data["text"], paragraph_id=data["paragraph_id"], name=data["name"])
        paragraphs[data["paragraph_id"]] = p

    # Load Events
    with open(save_root / "events.json", "r") as f:
        events_data = json.load(f)

    event_list = EventList()
    for data in events_data:
        agent = next(a for a in community if a.agent_id == data["agent_id"])
        paragraph = paragraphs[data["paragraph_id"]]
        vote = data["vote"]
        event = Event(a=agent, p=paragraph, v=vote)
        event_list.add_event(event)

    return community, event_list


def test_paper_example():

    ## Manually define an event list E:
    event_list = EventList()
    agents = [Agent(i) for i in range(1, 6)]  # a_1, a_2, a_3, a_4, a_5
    paragraphs = [Paragraph(paragraph_id=i, text=f"p{i}", name=f"p{i}") for i in range(1, 5)]  # p_1, p_2, p_3, p_4

    event_list.add_event(Event(a=agents[0], p=paragraphs[0], v=+1))  # (a_1, p_1, +1)
    event_list.add_event(Event(a=agents[1], p=paragraphs[0], v=+1))  # (a_2, p_1, +1)
    event_list.add_event(Event(a=agents[2], p=paragraphs[0], v=+1))  # (a_3, p_1, +1)

    event_list.add_event(Event(a=agents[0], p=paragraphs[1], v=+1))  # (a_1, p_2, +1)
    event_list.add_event(Event(a=agents[3], p=paragraphs[0], v=-1))  # (a_4, p_1, -1)
    event_list.add_event(Event(a=agents[4], p=paragraphs[0], v=-1))  # (a_5, p_1, -1)

    event_list.add_event(Event(a=agents[1], p=paragraphs[2], v=+1))  # (a_2, p_3, +1)
    event_list.add_event(Event(a=agents[0], p=paragraphs[2], v=+1))  # (a_1, p_3, +1)
    event_list.add_event(Event(a=agents[2], p=paragraphs[2], v=-1))  # (a_3, p_3, -1)

    event_list.add_event(Event(a=agents[2], p=paragraphs[3], v=+1))  # (a_3, p_4, +1)
    event_list.add_event(Event(a=agents[0], p=paragraphs[3], v=+1))  # (a_1, p_4, +1)
    event_list.add_event(Event(a=agents[4], p=paragraphs[3], v=-1))  # (a_5, p_4, -1)
    event_list.add_event(Event(a=agents[1], p=paragraphs[3], v=+1))  # (a_2, p_4, +1)
    event_list.add_event(Event(a=agents[3], p=paragraphs[3], v=+1))  # (a_4, p_4, +1)
    event_list.add_event(Event(a=agents[1], p=paragraphs[3], v=+0))  # (a_2, p_4, +0)
    event_list.add_event(Event(a=agents[3], p=paragraphs[2], v=-1))  # (a_4, p_3, -1)

    # Paragraph p1: Majority but tight (3+, 2-)
    # Paragraph p2: High consensus but very low participation (1+, 0-)
    # Paragraph p3: Exactly balanced votes (2+, 2-)
    # Paragraph p4: Slight majority but with agents having various influence (4+, 1-)

    # Global stance
    print(event_list.get_global_state())

    # State for each agent
    for a in agents:
        print(f"State for agent a{a.agent_id}\n{event_list.get_current_state_for_agent(a)}")

    # Static-CSF scores
    for p in paragraphs:
        print("\n")
        for r in rules:
            print(f"{r} score for paragraph {p.name}: {round(r.evaluate(E=event_list, p=p), 2)}")

    event_list.print_sum_matrix()
    event_list.print_tally_matrix()


if __name__ == "__main__":
    # test_event()
    # test_event_list()

    # Testing the construction of a saved event list
    # community, event_list = test_loading_eventlist()
    # print(event_list)

    test_paper_example()