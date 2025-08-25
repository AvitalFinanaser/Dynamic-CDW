# tests/test_scheduler.py
from typing import List
from events import *
from paragraphs import *
import datetime
import json
import os

from simulation import *
from agents import *
from rules import *
from pathlib import Path
from datasets.demographic_utils import sample_profiles, prepare_demographic_data


# Define configurations for examination

def get_test_configurations(num_agents=50, num_events=10, num_lists=2, num_paragraphs=100):
    """
    Returns a list of configurations for testing community creation.
    :param num_agents: Number of agents in each community
    :param num_events: Number of events in each list
    :param num_lists: Number of lists to generate
    :param num_paragraphs: Number of paragraphs in the context
    :return: A list of SimulationConfig objects
    """
    basic_context = ContextConfig(context_type="synthetic", num_paragraphs=num_paragraphs)

    return [
        # Test Case 1: Unstructured Agents
        SimulationConfig(
            num_agents=num_agents,
            num_events=num_events,
            num_lists=num_lists,
            agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
            context_config=basic_context
        ),
        # Test Case 2: Interval Agents with Uniform Distribution
        SimulationConfig(
            num_agents=num_agents,
            num_events=num_events,
            num_lists=num_lists,
            agent_config=AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=UniformDistribution()
            ),
            context_config=basic_context
        ),
        # Test Case 3: Interval Agents with Single-Peak Gaussian
        SimulationConfig(
            num_agents=num_agents,
            num_events=num_events,
            num_lists=num_lists,
            agent_config=AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=GaussianDistribution(mu=0.5, sigma=0.1)
            ),
            context_config=basic_context
        ),
        # Test Case 4: Interval Agents with Two-Peak Gaussian
        SimulationConfig(
            num_agents=num_agents,
            num_events=num_events,
            num_lists=num_lists,
            agent_config=AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=GaussianDistribution(mu=0.5, sigma=0.1),
                proportion_25=0.4
            ),
            context_config=basic_context
        ),
        # Test Case 5: LLM agents
        SimulationConfig(
            num_agents=num_agents,
            num_events=num_events,
            num_lists=num_lists,
            agent_config=AgentConfig(
                agent_type=AgentType.TEXTUAL,
            ),
            context_config=basic_context
        )
    ]


# Community

def analyze_community(community):
    """
    Analyzes and returns information about an agent community.
    This helps us understand what kind of agents were created and their properties.
    """
    # Get basic information about the community
    total_agents = len(community)
    agent_type = type(community[0]).__name__
    distribution_info = "None"

    # For interval agents, analyze their distributions
    if isinstance(community[0], AgentInterval):
        dist = community[0].distribution
        distribution_info = str(dist)

    return {
        "Community Size": total_agents,
        "Agent Type": agent_type,
        "Distribution": distribution_info,
    }


def print_community_agents(community):
    """
    Prints detailed information about each agent in the community
    using their str representation.
    """
    print("\nIndividual Agent Information:")
    print("-" * 50)
    for i, agent in enumerate(community, 1):
        print(f"Agent {i}: {str(agent)}")
    print("-" * 50)


def test_community_creation():
    """
    Tests creation of different types of agent communities.
    We'll create four different communities and verify their properties.
    """
    configurations = get_test_configurations()
    for idx, config in enumerate(configurations, 1):
        print(f"\n{idx}. Testing Configuration:")
        scheduler = Scheduler(config)
        community = scheduler.create_community()
        print(analyze_community(community))
        print_community_agents(community)


# Scheduling

def test_single_event_list_generation():
    """
    Tests the generation of a single event list for all configurations.
    For each configuration:
    1. Generate a single event list.
    2. Validate the structure and check the events_df method.
    """
    configurations = get_test_configurations()

    for idx, config in enumerate(configurations, 1):
        print(
            f"\nConfiguration {idx}, agents typed as {config.agent_config.agent_type}: Testing Single Event List Generation")
        scheduler = Scheduler(config)

        # Generate single event list
        community, event_list = scheduler.schedule_single_instance()

        # Print and validate event list
        print(f"Single Event List:")
        print(event_list.events_df())
        print(f"Community: {analyze_community(community)}")

    print("\nTest Passed: Single Event List Generation for All Configurations")


def test_multiple_event_lists_generation():
    """
    Tests the generation of multiple event lists for all configurations.
    For each configuration:
    1. Generate multiple event lists.
    2. Validate the structure of each event list and check the events_df method.
    """
    configurations = get_test_configurations()

    for idx, config in enumerate(configurations, 1):
        print(f"\nConfiguration {idx}: Testing Multiple Event Lists Generation")
        scheduler = Scheduler(config)

        # Generate multiple event lists
        multiple_results = scheduler.schedule_multiple_instances()
        print(multiple_results)
        # Validate and print each event list
        print(f"Multiple Event Lists for Configuration {idx}:")
        for list_idx, (community, event_list) in enumerate(multiple_results, 1):
            print(f"  Event List {list_idx}:")
            print(event_list.events_df())
            print(f"  Community {list_idx}: {analyze_community(community)}")

    print("\nTest Passed: Multiple Event Lists Generation for All Configurations")


def test_fixed_community_event_lists_generation():
    """
    Tests generating multiple event lists using a fixed community.
    Ensures that the community remains the same while events vary.
    """
    config = get_test_configurations(num_agents=5, num_events=20, num_lists=3)[3]
    scheduler = Scheduler(config)

    print("\nCreating Fixed Community...")
    fixed_community = scheduler.create_community()
    print(analyze_community(fixed_community))
    print_community_agents(fixed_community)

    print("\nGenerating Event Lists for Fixed Community...")
    event_lists = scheduler.schedule_multiple_instances_for_community(community=fixed_community, num_lists=3)

    for idx, event_list in enumerate(event_lists, 1):
        print(f"\nEvent List {idx}:")
        print(event_list.events_df())

        # Verify all agents match the fixed community
        agents_in_events = event_list.A_E()
        same_agents = all(a in fixed_community for a in agents_in_events)
        print_community_agents(agents_in_events)

        print(f"  All agents match fixed community: {same_agents}")
        assert same_agents, f"Event List {idx} contains agents not in fixed community!"

    print("\nTest Passed: Multiple Event Lists Generated from Fixed Community Correctly")


def full_schedule_eventlist_textual_script(num_Agents, num_Events, num_lists):
    num_Agents = num_Agents
    num_Events = num_Events
    num_lists = num_lists
    config = SimulationConfig(
        num_agents=num_Agents,
        num_events=num_Events,
        num_lists=num_lists,
        agent_config=AgentConfig(
            agent_type=AgentType.TEXTUAL,
        ),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=num_Events)
    )
    schedule = Scheduler(config)
    agents = schedule.create_community()
    event_list = Scheduler.schedule_eventlist_textual(agents=agents, num_events=num_Events)
    return event_list


def test_schedule_eventlist_textual():
    # 1. Prepare Data & Community
    # start = datetime.datetime.now()
    num_Agents = 5
    num_Events = 20
    num_lists = 1
    # df_prepared = prepare_demographic_data()
    # sampled_profiles = sample_profiles(df_prepared=df_prepared, num_samples=num_Agents)
    # agents = LLMAgent.create_community(sampled_profiles, topic="climate change policy")
    # end = datetime.datetime.now() - start
    # print(f"Creation time for {num_Agents} agents: {end}")
    # start = datetime.datetime.now()

    # 1.1 Community via scheduler instance
    config = SimulationConfig(
        num_agents=num_Agents,
        num_events=num_Events,
        num_lists=num_lists,
        agent_config=AgentConfig(
            agent_type=AgentType.TEXTUAL,
        ),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=num_Events),
        random_seed=43
    )
    schedule = Scheduler(config)
    agents = schedule.create_community()
    print_community_agents(agents)

    # 1.2 Communities creation
    # schedule.config.num_lists = 2
    # communities = schedule.create_communities()
    # print(communities)
    # for i, agents in enumerate(communities):
    #     print(f"Community {i + 1}:\n {[str(agent) for agent in agents]}")

    # 2.A Generate Event List without Rule
    # print(f"Starting scheduling at {datetime.datetime.now()}")
    # event_list = Scheduler.schedule_eventlist_textual(agents=agents, num_events=num_Events)
    # end = datetime.datetime.now() - start
    # print(f"Creation time for {num_Events} events: {end}")

    # 2.B Generate an Event List with Rule
    condition = StaticCondition(CSF="APS", threshold=0.7, beta=0.05)
    # condition = SmoothDynamicCondition(
    #     CSF="APS",
    #     threshold=0.7,
    #     dynamic_property=EventsDynamicProperty(),
    #     F=ExpSmoothingFunction(alpha=0.3),
    #     beta=0.1
    # )
    event_list = Scheduler.schedule_eventlist_textual_Rule(agents=agents, num_events=num_Events, rule=condition)
    print(event_list)
    print(event_list.get_global_state())
    sol = condition.solution_1Condition(event_list)
    for p in sol:
        print(f"{p._name}: {p._text}")
    return agents, event_list

    # # 3. Print Results
    # print("\n=== Event List Summary ===")
    # print(event_list)
    #
    # print("\n=== Detailed Events ===")
    # print((event_list.get_global_state()))

    # 4. Evaluate metrics
    # system = CDWSystem(scheduler=schedule)
    # system.analyze_multiple_metrics(condition)


def test_saving_event_list():
    # conf
    num_Agents = 5
    num_Events = 4
    num_lists = 1
    config = SimulationConfig(
        num_agents=num_Agents,
        num_events=num_Events,
        num_lists=num_lists,
        agent_config=AgentConfig(
            agent_type=AgentType.TEXTUAL,
        ),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=num_Events)
    )
    # generate (A, E)
    schedule = Scheduler(config)
    agents = schedule.create_community()
    condition = StaticCondition(CSF="AM", threshold=0.5, beta=0.05)
    event_list = Scheduler.schedule_eventlist_textual_Rule(agents=agents, num_events=num_Events, rule=condition)
    print_community_agents(agents)
    print(event_list)
    # JSON files creation
    # 1- agents
    agents_data = [
        {
            "agent_id": agent.agent_id,
            "profile": agent._profile,  # Already in correct structure
            "topic": agent._topic,
            "topic_position": agent._topic_position
        }
        for agent in agents
    ]
    print(agents_data)
    # 2 - paragraphs
    paragraphs_data = [
        {
            "paragraph_id": p._paragraph_id,
            "text": p._text,
            "name": p._name
        }
        for p in event_list.P()
    ]
    print(paragraphs_data)
    # 3 - events
    events_data = [
        {
            "agent_id": event.agent.agent_id,
            "paragraph_id": event.paragraph.paragraph_id,
            "vote": event.vote
        }
        for event in event_list.events
    ]
    print(events_data)
    # Saving
    save_root = Path("datasets/event_lists")
    save_root.mkdir(parents=True, exist_ok=True)
    # 1 Create New Folder Inside event_lists (Auto-create event_list_N)
    existing_folders = [d for d in save_root.iterdir() if d.is_dir() and d.name.startswith("event_list_")]
    next_index = len(existing_folders) + 1
    save_dir = save_root / f"event_list_{next_index}"
    save_dir.mkdir()
    # 2 saving as JSON
    with open(save_dir / "agents.json", "w") as f:
        json.dump(agents_data, f, indent=2)

    with open(save_dir / "paragraphs.json", "w") as f:
        json.dump(paragraphs_data, f, indent=2)

    with open(save_dir / "events.json", "w") as f:
        json.dump(events_data, f, indent=2)

    print(f"Saved event list to: {save_dir}")


def save_event_list(agents: List[LLMAgent], event_list: EventList):
    agents_data = [
        {
            "agent_id": agent.agent_id,
            "profile": agent.profile,
            "topic": agent.topic,
            "topic_position": agent.topic_position
        } for agent in agents
    ]

    paragraphs_data = [
        {
            "paragraph_id": p.paragraph_id,
            "text": p.text,
            "name": p.name
        } for p in event_list.P()
    ]

    events_data = [
        {
            "agent_id": event.agent.agent_id,
            "paragraph_id": event.paragraph.paragraph_id,
            "vote": event.vote
        } for event in event_list.events
    ]

    save_root = Path("datasets/event_lists/example")
    save_root.mkdir(parents=True, exist_ok=True)

    existing_folders = [d for d in save_root.iterdir() if d.is_dir() and d.name.startswith("event_list_")]
    next_index = len(existing_folders) + 1
    save_dir = save_root / f"event_list_{next_index}"
    save_dir.mkdir()

    with open(save_dir / "agents.json", "w") as f:
        json.dump(agents_data, f, indent=2)

    with open(save_dir / "paragraphs.json", "w") as f:
        json.dump(paragraphs_data, f, indent=2)

    with open(save_dir / "events.json", "w") as f:
        json.dump(events_data, f, indent=2)


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
    # test_community_creation()
    # test_single_event_list_generation()
    # test_multiple_event_lists_generation()
    # test_fixed_community_event_lists_generation()
    # test_saving_event_list()
    # E = full_schedule_eventlist_textual_script(num_Agents=20, num_Events=150, num_lists=5)
    agents, E = test_schedule_eventlist_textual()
    save_event_list(agents, E)

    # Loading the event list
    E = load_event_list(event_list_path=Path("datasets/event_lists/example/event_list_4"))
    print(E)

    print(E.get_global_state())

    # For event 38
    E_i = EventList()

    condition = StaticCondition(CSF="APS", threshold=0.7)
    solution = condition.solution_1Condition(E=E)
    print(solution)

    for p in solution:
        print(f"\n{p}")

    for a in E.A_E():
        print(f"\n{a}")
    counts = {}
    # Count occurrences of each category
    for a in E.A_E():
        category = a._position_category
        if category in counts:
            counts[category] += 1
        else:
            counts[category] = 1
    # Print each category and its count
    for category, count in counts.items():
        print(f"{category}: {count}")

