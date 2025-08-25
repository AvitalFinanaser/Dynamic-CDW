# The scheduler is responsible for managing event list generation and handling in our simulation.
import multiprocessing
from datetime import datetime
from simulation.config import SimulationConfig, AgentConfig, ContextConfig, AgentType
from agents import *
from events import *
from paragraphs import Paragraph, ParagraphWithScore
from rules import *
from datasets.demographic_utils import sample_profiles, prepare_demographic_data
from typing import List
import random


# Help functions to distinguish between the agents models

# def schedule_eventlist_unstructured(agents: List[Agent], num_events: int) -> EventList:
#     """
#     Creates a single event list based on configuration settings suited for unstructured community.
#     :param agents: Community of agents involve in the event list
#     :param num_events: the length of the event list generated
#     :return: Event list event_list
#     """
#     event_list = EventList()
#     valid_event_count = 0
#
#     while valid_event_count < num_events:
#         a = random.choice(agents)  # random choice of an agent
#
#         # random choice of an event: 0 - new proposal, 1 - new vote, 2 - update vote
#         event_type = random.randint(0, 2)
#
#         if event_type == 0:  # 0 - New paragraph suggestion
#             # Generating a new paragraph
#             v = +1
#             paragraph_id = len(event_list.P()) + 1
#             available_paragraphs = [
#                 Paragraph(f"p{paragraph_id}", paragraph_id, f"p{paragraph_id}")]  # New paragraph creation
#         elif event_type == 1:  # 1 - New vote on an existing paragraph
#             # P(e)
#             v = random.choice([+1, -1])
#             # Available paragraphs are those with no vote on (get vote returns -2)
#             available_paragraphs = [p for p in event_list.P() if event_list.get_vote(a, p) == -2]
#         else:  # 2 - Update vote
#             # P(e) | a
#             v = 0
#             # paragraphs with last vote from agent different from 0
#             available_paragraphs = [p for p in event_list.P() if event_list.get_vote(a, p) in [-1, 1]]
#
#         if not available_paragraphs:
#             continue  # Skip if no valid paragraph is available
#
#         p = random.choice(available_paragraphs)  # Random choice of paragraph
#         event = Event(a, p, v)
#         event_list.add_event(event)
#         valid_event_count += 1  # Only count valid events
#
#     return event_list
#
#
# def schedule_eventlist_euclidean(agents: List[Agent], num_events: int) -> EventList:
#     """
#     Creates a single event list based on configuration settings suited for Euclidean community.
#     :param agents: Community of agents involve in the event list
#     :param num_events: the length of the event list generated
#     :return: an event list E where the num_events events are scheduled by num_events from distribution
#     """
#     event_list = EventList()
#
#     # Creating num_events events
#     for _ in range(num_events):
#         # random choice of an agent
#         a = random.choice(agents)
#
#         # Calculate the ratio of proposed paragraphs within the agent's interval
#         proposed_paragraphs_in_interval = [p for p in event_list.P() if a.min_score <= p.score <= a.max_score]
#         total_proposed_paragraphs = len(event_list.P())
#         ratio = len(proposed_paragraphs_in_interval) / total_proposed_paragraphs if total_proposed_paragraphs > 0 else 0
#
#         # Determine if the agent should propose a new paragraph
#         if ratio < 0.2:
#             # Propose a new paragraph within the agent's interval
#             event = Event(a, a.propose_paragraph(event_list), +1)
#         else:
#             # Vote on an existing paragraph
#             available_paragraphs = event_list.P()
#             if not available_paragraphs:
#                 # If there are no existing paragraphs then a new one is suggested
#                 event = Event(a, a.propose_paragraph(event_list), +1)
#             else:
#                 # There is at least one paragraph
#                 paragraph = random.choice(available_paragraphs)  # Random choice of a paragraph suggested
#                 current_vote = event_list.get_vote(a, paragraph)  # Receive the last vote of the agent on p
#                 if current_vote in [1, -1]:
#                     # Change vote to 0 if already voted
#                     vote = 0
#                 else:
#                     # Vote based on the paragraph score relative to the agent's interval
#                     vote = a.vote_on_paragraph(paragraph)
#                 event = Event(a, paragraph, vote)
#
#         event_list.add_event(event)
#
#     return event_list


class Scheduler:
    """
    Manages event generation and order of events (event list) simulation.
    Works with the configuration to create appropriate event lists and handles the sequencing of events.
    """

    def __init__(self, config: SimulationConfig):
        # Step 1 : system configuration
        self._config = config
        # Step 2: set random
        random.seed(self._config.random_seed)
        # Step 3: generate relevant agents context - demographic and on-topic examples
        self._demographic = prepare_demographic_data()

    # Step 4: Create community A
    def create_community(self) -> list:
        """
        Creates a community of agents based on the configuration settings.

        The function handles multiple agent types:
        1. Unstructured Agent: Regular agent without scoring mechanism
        2. AgentInterval with UniformDistribution: Agents using uniform intervals
        3. AgentInterval with GaussianDistribution: Agents using Gaussian intervals
           - Single peak: All agents use same Gaussian parameters
           - Two peaks: Agents split between two Gaussian centers (when proportion_25 > 0)

        For UNSTRUCTURED type, creates basic agents without intervals.
        For INTERVAL type, creates agents with the specified distribution.
        """
        agent_type = self._config.agent_config.agent_type
        num_agents = self._config.num_agents
        distribution = self._config.agent_config.distribution
        proportion_25 = self._config.agent_config.proportion_25

        # Unstructured agent type - these are basic agents without intervals
        if agent_type == AgentType.UNSTRUCTURED:
            return [Agent(agent_id=i) for i in range(1, num_agents + 1)]

        # Interval agent type - Euclidean interval in [0,1] following a distribution
        if agent_type == AgentType.INTERVAL:
            # # Two peaks Gaussian - if we have a Gaussian with proportion_25
            if isinstance(distribution, GaussianDistribution) and proportion_25 > 0:
                num_agents_25 = int(num_agents * proportion_25)
                return [
                    AgentInterval(
                        agent_id=i,
                        distribution=GaussianDistribution(
                            mu=0.25 if i <= num_agents_25 else 0.75,
                            sigma=distribution.sigma
                        )
                    ) for i in range(1, num_agents + 1)
                ]

            # Single distribution case (either Uniform or single-peak Gaussian according to distribution)
            return [
                AgentInterval(agent_id=i, distribution=distribution)
                for i in range(1, num_agents + 1)
            ]
        if agent_type == AgentType.TEXTUAL:
            sampled_profiles = sample_profiles(df_prepared=self._demographic, num_samples=num_agents)
            return LLMAgent.create_community(sampled_profiles, topic=self._config.context_config.topic)

        raise ValueError(f"Unknown agent type: {agent_type}")

    def create_communities(self):
        communities = []
        base_seed = self._config.random_seed

        for instance_id in range(self._config.num_lists):
            random.seed(base_seed + instance_id)  # distinct seed per community
            agents = self.create_community()
            communities.append(agents)

        return communities

    # Step 5: schedule a single event list using (A,P) -> E
    def schedule_single_instance(self) -> tuple[List['Agent'], 'EventList']:
        """Creates a single event list based on configuration settings."""
        """
        Creates a single event list based on configuration settings:
        :param num_agents: Number of agents in community - we create a community
        :param num_events: Number of events in each list    
        :return: A tuple containing:
                - The list of agents in the community
                - The generated event list        
        """
        # Step 1: Create the community
        community = self.create_community()
        # Step 2: Generate the event list based on the agent type
        # Unstructured
        if self._config.agent_config.agent_type == AgentType.UNSTRUCTURED:
            event_list = self.schedule_eventlist_unstructured(
                agents=community,
                num_events=self._config.num_events
            )
        elif self._config.agent_config.agent_type == AgentType.TEXTUAL:
            event_list = self.schedule_eventlist_textual(
                agents=community,
                num_events=self._config.num_events
            )
        else:
            # Else - Euclidean
            event_list = self.schedule_eventlist_euclidean(
                agents=community,
                num_events=self._config.num_events
            )
        return community, event_list

    # Step 6 (1): schedule a multiple event lists with (A, P)_i -> E
    def schedule_multiple_instances(self) -> List[tuple[List[Agent], EventList]]:
        """
        Main orchestration method that handles the complete process multiple event lists:
        1. Creating multiple communities
        2. Generating event lists for each community
        3. Managing random seeds throughout the process

        :return: A list of tuples, where each tuple contains:
                 - The list of agents in the community
                 - The corresponding event list for that community

        """
        original_state = random.getstate()
        results = []

        # Generate multiple communities and their event lists - instances
        for list_index in range(self._config.num_lists):
            # Create new seeds for this iteration
            list_seed = self._config.random_seed + list_index
            random.seed(list_seed)

            # Use schedule_single_instance to generate the community and event list
            community, event_list = self.schedule_single_instance()

            # Append the result as a tuple
            results.append((community, event_list))

        # Restore original random state
        random.setstate(original_state)
        return results

    # Step 6 (2): schedule a multiple event lists given a certain community (A, P) -> E
    def schedule_multiple_instances_for_community(self, community: List[Agent], num_lists: int) -> List[EventList]:
        """
        Generates multiple event lists for a fixed community.

        Args:
            community: A list of agents to use for all event lists - [Agent].
            num_lists: The number of event lists to generate.

        Returns:
            List of EventLists generated for the same community.
        """
        original_state = random.getstate()
        event_lists = []

        # For each event list
        for list_index in range(num_lists):
            # set seed for reconstruction
            list_seed = self._config.random_seed + list_index
            random.seed(list_seed)

            # generate event list for this fixed community
            if self._config.agent_config.agent_type == AgentType.UNSTRUCTURED:
                event_list = self.schedule_eventlist_unstructured(
                    agents=community,
                    num_events=self._config.num_events
                )
            elif self._config.agent_config.agent_type == AgentType.TEXTUAL:
                event_list = self.schedule_eventlist_textual(
                    agents=community,
                    num_events=self._config.num_events
                )
            else:
                event_list = self.schedule_eventlist_euclidean(
                    agents=community,
                    num_events=self._config.num_events
                )

            event_lists.append(event_list)

        random.setstate(original_state)
        return event_lists

    @staticmethod
    def schedule_eventlist_textual(agents: List[Agent], num_events: int) -> EventList:
        """
        Creates a single event list based on configuration settings suited for LLM agents community.
        :param agents: Community of agents involve in the event list
        :param num_events: the length of the event list generated
        :return: an event list E where the num_events events are scheduled by num_events from distribution
        """
        event_list = EventList()

        # Creating num_events events
        for _ in range(num_events):
            # random choice of an agent
            a = random.choice(agents)

            # retrieve current state
            current_state = event_list.get_current_state_for_prompt(a)

            # agent's event generation
            action = a.decision_chat(current_state)
            event = event_list.event_from_action(a, action)
            event_list.add_event(event)

        return event_list

    @staticmethod
    def schedule_eventlist_textual_Rule(agents: List[Agent], num_events: int, rule: Condition) -> EventList:
        """
        Creates a single event list based on configuration settings suited for LLM agents community with solution view.
        :param agents: Community of agents involve in the event list
        :param num_events: The length of the event list generated
        :param rule: CCR
        :return: An event list E where the num_events events are scheduled
        """
        event_list = EventList()

        # Creating num_events events
        for _ in range(num_events):
            # random choice of an agent
            a = random.choice(agents)
            start_time = datetime.now()

            # Current solution
            solution = rule.solution_1Condition(E=event_list)

            # retrieve current state
            current_state = event_list.get_current_state_solution_for_prompt(a, solution)

            # agent's event generation
            action = a.decision_chat(current_state)
            event = event_list.event_from_action(a, action)
            print(f"{event} : {event.paragraph._text}")
            event_list.add_event(event)
            worker_name = multiprocessing.current_process().name
            end_time = datetime.now()
            print(f"{worker_name} [END SCHEDULING EVENT {len(event_list.events)} at {end_time} => took{(datetime.now() - start_time)}]", flush=True)


        return event_list


    @staticmethod
    def schedule_eventlist_unstructured(agents: List[Agent], num_events: int) -> EventList:
        """
        Creates a single event list based on configuration settings suited for unstructured community.
        :param agents: Community of agents involve in the event list
        :param num_events: the length of the event list generated
        :return: Event list event_list
        """
        event_list = EventList()
        valid_event_count = 0

        while valid_event_count < num_events:
            a = random.choice(agents)  # random choice of an agent

            # random choice of an event: 0 - new proposal, 1 - new vote, 2 - update vote
            event_type = random.randint(0, 2)

            if event_type == 0:  # 0 - New paragraph suggestion
                # Generating a new paragraph
                v = +1
                paragraph_id = len(event_list.P()) + 1
                available_paragraphs = [
                    Paragraph(f"p{paragraph_id}", paragraph_id, f"p{paragraph_id}")]  # New paragraph creation
            elif event_type == 1:  # 1 - New vote on an existing paragraph
                # P(e)
                v = random.choice([+1, -1])
                # Available paragraphs are those with no vote on (get vote returns -2)
                available_paragraphs = [p for p in event_list.P() if event_list.get_vote(a, p) == -2]
            else:  # 2 - Update vote
                # P(e) | a
                v = 0
                # paragraphs with last vote from agent different from 0
                available_paragraphs = [p for p in event_list.P() if event_list.get_vote(a, p) in [-1, 1]]

            if not available_paragraphs:
                continue  # Skip if no valid paragraph is available

            p = random.choice(available_paragraphs)  # Random choice of paragraph
            event = Event(a, p, v)
            event_list.add_event(event)
            valid_event_count += 1  # Only count valid events

        return event_list

    @staticmethod
    def schedule_eventlist_euclidean(agents: List[Agent], num_events: int) -> EventList:
        """
        Creates a single event list based on configuration settings suited for Euclidean community.
        :param agents: Community of agents involve in the event list
        :param num_events: the length of the event list generated
        :return: an event list E where the num_events events are scheduled by num_events from distribution
        """
        event_list = EventList()

        # Creating num_events events
        for _ in range(num_events):
            # random choice of an agent
            a = random.choice(agents)

            # Calculate the ratio of proposed paragraphs within the agent's interval
            proposed_paragraphs_in_interval = [p for p in event_list.P() if a.min_score <= p.score <= a.max_score]
            total_proposed_paragraphs = len(event_list.P())
            ratio = len(
                proposed_paragraphs_in_interval) / total_proposed_paragraphs if total_proposed_paragraphs > 0 else 0

            # Determine if the agent should propose a new paragraph
            if ratio < 0.2:
                # Propose a new paragraph within the agent's interval
                event = Event(a, a.propose_paragraph(event_list), +1)
            else:
                # Vote on an existing paragraph
                available_paragraphs = event_list.P()
                if not available_paragraphs:
                    # If there are no existing paragraphs then a new one is suggested
                    event = Event(a, a.propose_paragraph(event_list), +1)
                else:
                    # There is at least one paragraph
                    paragraph = random.choice(available_paragraphs)  # Random choice of a paragraph suggested
                    current_vote = event_list.get_vote(a, paragraph)  # Receive the last vote of the agent on p
                    if current_vote in [1, -1]:
                        # Change vote to 0 if already voted
                        vote = 0
                    else:
                        # Vote based on the paragraph score relative to the agent's interval
                        vote = a.vote_on_paragraph(paragraph)
                    event = Event(a, paragraph, vote)

            event_list.add_event(event)

        return event_list

    @property
    def config(self):
        # Getter for scheduler config
        return self._config

    # Help methods
    @config.setter
    def config(self, value):
        self._config = value