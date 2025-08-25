from datetime import datetime

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List
from simulation import *
from events import *
from agents import *
from paragraphs import *
from agents import *
from rules import *
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def load_event_list(event_list_path: str):
    # Load the event list components
    with open(event_list_path / "agents.json") as f:
        agents_data = json.load(f)
    with open(event_list_path / "events.json") as f:
        events_data = json.load(f)
    with open(event_list_path / "paragraphs.json") as f:
        paragraphs_data = json.load(f)

    # Construct agents
    community = []
    for data in agents_data:
        agent = LLMAgent(
            agent_id=data["agent_id"],
            profile=data["profile"],
            topic=data["topic"],
            topic_position=data["topic_position"]
        )
        community.append(agent)

    # Construct paragraph
    paragraphs = {}
    for data in paragraphs_data:
        p = Paragraph(text=data["text"], paragraph_id=data["paragraph_id"], name=data["name"])
        paragraphs[data["paragraph_id"]] = p

    # Construct event list
    event_list = EventList()
    for data in events_data:
        agent = next(a for a in community if a.agent_id == data["agent_id"])
        paragraph = paragraphs[data["paragraph_id"]]
        vote = data["vote"]
        event = Event(a=agent, p=paragraph, v=vote)
        event_list.add_event(event)

    return event_list


if __name__ == "__main__":
    event_list_path = Path("datasets/event_lists/config002_llm/(CSF=APS,_Function=linear_alpha=0.15_events,_threshold=0.5)/instance_0")
    E = load_event_list(event_list_path=event_list_path)
    rule = SmoothDynamicCondition(CSF="APS", threshold=0.5, dynamic_property=EventsDynamicProperty(), F=LinearSmoothingFunction(alpha=0.15))
    solution = rule.solution_1Condition(E=E)
    for p in solution:
        print(f"\n{p}")
    print(E.get_global_state())
    print(solution)
    #
    # counts = {}
    # # Count occurrences of each category
    # for a in E.A_E():
    #     category = a._position_category
    #     if category in counts:
    #         counts[category] += 1
    #     else:
    #         counts[category] = 1
    # # Print each category and its count
    # for category, count in counts.items():
    #     print(f"{category}: {count}")
    #
    # E_i = EventList()
    # E_i.events = E.E_i(50)
    # for p in E_i.P():
    #     print(p.text)

    #"""
    config = SimulationConfig(
        num_agents=20,
        num_events=100,
        num_lists=1,
        agent_config=AgentConfig(agent_type=AgentType.TEXTUAL),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=100),
        random_seed=42
    )
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)
    rule = StaticCondition("APS_r", 0.7)
    agents = Scheduler.create_community(scheduler)
    E = Scheduler.schedule_eventlist_textual_Rule(agents=agents, num_events=50, rule=rule)
