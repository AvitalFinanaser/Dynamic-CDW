import json

from events.event import Event
from paragraphs import *
from typing import List, Dict
from tabulate import tabulate
import pandas as pd
from agents import *


class EventList:
    """
    Represents a list of events in the system.
    Attributes:
        _events (List[Event]): The list of events
    """

    def __init__(self, events: List[Event] = None):
        """
        Initialize an EventList instance.
        """
        self._events = events if events else []

    def add_event(self, event):
        """
        Add an event to the event list.
        :param event: The event to add
        """
        self._events.append(event)

    def p_plus(self, p):
        """
        Calculate the total up-votes (p^+) that a paragraph p gained in E, considering only the last vote.
        :param p: The Paragraph object.
        :return: The total up-votes for the paragraph p.
        """
        count = 0
        agents = set(event.agent for event in self._events)
        for agent in agents:
            if self.get_vote(agent, p) == +1:
                count += 1
        return count

    def p_minus(self, p):
        # (p^-) - Total down-votes paragraph p gained in E
        """
        Calculate the total down-votes (p^-) that a paragraph p gained in E, considering only the last vote.
        :param p: The Paragraph object.
        :return: The total down-votes for the paragraph p.
        """
        count = 0
        agents = set(event.agent for event in self._events)
        for agent in agents:
            if self.get_vote(agent, p) == -1:
                count += 1
        return count

    def p_plus_r(self, p):
        """
        Calculate the total relative up-votes (p^+_r) that a paragraph p gained in E, considering only the last vote.
        :param p: The Paragraph object.
        :return: The total relative up-votes for the paragraph p = sum_a (1/Na is V(a,p) = 1)
        """
        score = 0
        agents = set(event.agent for event in self._events)
        for agent in agents:
            if self.get_vote(agent, p) == +1:
                Na = self.agentNa(agent)
                score += 1 / Na
        return score

    def p_minus_r(self, p):
        """
        Calculate the total relative down-votes (p^-_r) that a paragraph p gained in E, considering only the last vote.
        :param p: The Paragraph object.
        :return: The total relative down-votes for the paragraph p = sum_a (1/Na is V(a,p) = 1)
        """
        score = 0
        agents = set(event.agent for event in self._events)
        for agent in agents:
            if self.get_vote(agent, p) == -1:
                Na = self.agentNa(agent)
                score += 1 / Na
        return score

    def APS_r(self, p: Paragraph) -> float:
        """
        APS_r:= p^+_r / p^+_r + p^-_r
        :param p: Paragraph
        :return: APS_r consensus score for p given E
        """
        # Approval Proportion Score
        score = 0.0
        p_plus_r = 0
        p_minus_r = 0
        agents = set(event.agent for event in self._events)
        for agent in agents:
            if self.get_vote(agent, p) == -1:
                Na = self.agentNa(agent)
                p_minus_r += 1 / Na
            else:
                if self.get_vote(agent, p) == +1:
                    Na = self.agentNa(agent)
                    p_plus_r += 1 / Na
        total_relative = p_plus_r + p_minus_r
        if p in self.P() and self.events and total_relative != 0:  # Input check: p in P(E), else returns 0
            score = p_plus_r / total_relative
        return score

    def p_avoid(self, p):
        # (p^0) - Total avoid-votes paragraph p gained in E. Only actually voted not empty votes.
        count = 0
        for e in self._events:
            if e.paragraph == p and e.vote == 0:
                count = count + 1
        return count

    def p_by_id(self, p_id: int) -> 'Paragraph':
        # paragraph id -> paragraph
        for e in self._events:
            p = e.paragraph
            if p.paragraph_id == p_id:
                return p
        raise ValueError(f"Paragraph with ID {p_id} not found in EventList.")

    def P(self):
        # P(E) - Returns the list of paragraphs in E
        """
        List of paragraphs in E
        :return: List of paragraph objects
        """
        paragraphs = []
        for e in self._events:
            if e.paragraph not in paragraphs:
                paragraphs.append(e.paragraph)
        return paragraphs

    def E_p(self, p):
        # (E_p) - sub list of the events on paragraph p
        """
        sub list of the events on paragraph p
        :param p: Paragraph
        :return: Event list - E_p
        """
        events_p = []
        for e in self._events:
            if e.paragraph == p:
                events_p.append(e)
        return events_p

    def E_a(self, a):
        # (E_a) - sub list of the events made by agent a
        """
        Sub-eventlist of the events made by agent a
        :param a: Agent
        :return: Event list - E_a
        """
        events_a = []
        for e in self._events:
            if e.agent == a:
                events_a.append(e)
        return events_a

    def E_i(self, i: int):
        """
        Sub-eventlist of the first i events
        :param i: Number of events the sub-list should contain
        :return: Event list - E_i
        """
        # (E_i) = sub list of events until event i
        if i < 0 or i > len(self._events):  # input i check
            raise IndexError("Index i out of range in method E_i")
        return self._events[:i].copy()

    def A_p(self, p):
        """
        Sub list of the agents who generated an event on paragraph p
        :param p: Paragraph
        :return: Agents list A_p
        """
        events_a_p = []
        for e in self._events:
            if e.paragraph == p:
                events_a_p.append(e.agent)
        return events_a_p

    def A_E(self):
        # (A_E) - sub list of the agents who generated an event on in event list E
        events_a_e = []
        for e in self._events:
            if e.agent not in events_a_e:
                events_a_e.append(e.agent)
        return events_a_e

    @property
    def events(self) -> List[Event]:
        # Return a copy of the events list to prevent external modification
        return self._events.copy()

    @events.setter
    def events(self, another_list: List[Event]):
        if isinstance(another_list, list):
            self._events = another_list
        else:
            raise ValueError("Events must be a list")

    def events_df(self):
        """
        Creates a pandas DataFrame of the events
        :return: DataFrame containing the events
        """
        # Extract event data
        data = {
            'Index': list(range(1, len(self.events) + 1)),
            'Agent': [event.agent.name for event in self.events],
            'Paragraph': [event.paragraph.name for event in self.events],
            'Vote': [event.vote for event in self.events]
        }

        # Create DataFrame
        df = pd.DataFrame(data)
        return df

    def get_vote(self, a: Agent, p: Paragraph):
        """
        Get the current vote of an agent a on paragraph p
        :param a: The agent whose vote is to be retrieved.
        :param p: The paragraph for which the vote is to be retrieved.
        :return: The vote of the agent for the paragraph, or -2 if no vote is found.
        """
        for event in self._events[::-1]:  # Traverse events in reverse order to get the latest vote
            if event.agent == a and event.paragraph == p:
                return event.vote
        # No vote was found
        return -2

    def sum(self):
        """
        Create a summary matrix of the last votes for each agent on each paragraph.
        Rows correspond to agents, columns correspond to paragraphs.
        Note - to address cell [i,j] then  use agents, paragraphs, summary_matrix = event_list.sum() and then summary_matrix[i],[j] starting from 0 index
        :return: A tuple containing lists of agent IDs, paragraph IDs, and the summary matrix.
        """
        # Get a sorted list of unique agent and paragraph IDs
        agents = sorted({event.agent.agent_id for event in self._events})
        paragraphs = sorted({event.paragraph.paragraph_id for event in self._events})

        # Create index mappings for agents and paragraphs
        agent_index = {agent_id: i for i, agent_id in enumerate(agents)}
        paragraph_index = {paragraph_id: j for j, paragraph_id in enumerate(paragraphs)}

        # Initialize the matrix with zeros
        summary_matrix = [[0 for _ in paragraphs] for _ in agents]

        # Fill the matrix with the last vote of each agent on each paragraph
        for event in self._events:
            i = agent_index[event.agent.agent_id]
            j = paragraph_index[event.paragraph.paragraph_id]
            summary_matrix[i][j] = self.get_vote(event.agent, event.paragraph)

        return agents, paragraphs, summary_matrix

    def print_sum_matrix(self):
        agents, paragraphs, summary_matrix = self.sum()

        # Create table headers and data
        table_headers = ['Agent/Paragraph'] + [Paragraph.get_paragraph_name_by_id(paragraph_id, self.P()) for
                                               paragraph_id in paragraphs]
        table_data = [[Agent.get_agent_name_by_id(agent_id, self.A_E())] + row for agent_id, row in
                      zip(agents, summary_matrix)]

        # Print table
        print(tabulate(table_data, headers=table_headers, tablefmt="grid", stralign="center"))
        return table_data

    def tally(self):
        """
        Tally matrix of event list E, with rows representing the different types of votes (positive or negative)
         and columns representing the different paragraphs.

        :return: A Pandas DataFrame representing the tally matrix.
        """

        # Initialize the tally dictionary
        paragraphs = self.P()
        paragraph_names = {p.name for p in paragraphs}
        sorted_paragraph_names = sorted(paragraph_names, key=lambda x: int(x[1:]))  # Extract numeric part after 'p'
        tally_matrix = {paragraph_name: {'p+': 0, 'p-': 0} for paragraph_name in sorted_paragraph_names}

        # Iterate the paragraphs
        for p in paragraphs:
            paragraph_name = p.name
            p_plus = self.p_plus(p)
            p_minus = self.p_minus(p)
            tally_matrix[paragraph_name]['p-'] = p_minus
            tally_matrix[paragraph_name]['p+'] = p_plus

        # Creating a Pandas Dataframe
        tally_df = pd.DataFrame(tally_matrix)

        return tally_df

    def tally_r(self):
        """
        Tally relative matrix of event list E, with rows representing the different types of relative votes (p^+_r or p^-_r)
        and columns representing the different paragraphs.
        :return: A Pandas DataFrame representing the tally relative matrix.
        """

        # Initialize the tally dictionary
        paragraphs = self.P()
        paragraph_names = {p.name for p in paragraphs}
        sorted_paragraph_names = sorted(paragraph_names, key=lambda x: int(x[1:]))  # Extract numeric part after 'p'
        tally_matrix = {paragraph_name: {'p+r': 0, 'p-r': 0} for paragraph_name in sorted_paragraph_names}

        # Iterate the paragraphs
        for p in paragraphs:
            paragraph_name = p.name
            p_plus = self.p_plus_r(p)
            p_minus = self.p_minus_r(p)
            tally_matrix[paragraph_name]['p-r'] = p_minus
            tally_matrix[paragraph_name]['p+r'] = p_plus

        # Creating a Pandas Dataframe
        tally_df = pd.DataFrame(tally_matrix)

        return tally_df

    def print_tally_matrix(self):
        """
        Print of tally matrix as a table
        """
        tally_df = self.tally()

        # Print table
        print(tabulate(tally_df, headers='keys', tablefmt="grid", stralign="center"))
        return tally_df

    def activeAgents(self):
        """
        Returns a list of agents that have cast at least one non-zero vote across all paragraphs.
        :return: A list of agent IDs with at least one non-zero vote.
        """
        # Get the summary matrix from the sum method
        agents, paragraphs, summary_matrix = self.sum()
        agent_objects = self.A_E()  # Get the agent objects from the events list
        active_agents = []

        # Create a mapping of agent IDs to agent objects
        agent_id_to_object = {agent.agent_id: agent for agent in agent_objects}

        # Iterate through the summary matrix to find agents with non-zero votes
        for i, agent_votes in enumerate(summary_matrix):
            # If any vote for this agent is non-zero, they are considered active
            if any(vote != 0 for vote in agent_votes):
                agent_id = agents[i]  # Get the agent ID
                if agent_id in agent_id_to_object:
                    active_agents.append(agent_id_to_object[agent_id])

        return active_agents

    def __str__(self):
        """
        Returns a string representation of the EventList.
        """
        if not self.events:
            return "EventList is empty"
        return "\n".join([f"e{i + 1} = {str(event)}" for i, event in enumerate(self.events)])

    def equals(self, events2: 'EventList') -> bool:
        return self._events.__eq__(events2.events)

    def agentNa(self, agent):
        """
        Number of total preferences votes of agent in Event list.
        :param agent: The agent object.
        :return: Na = TP + TN + FP + FN
        """
        Na = 0
        for p in self.P():
            if self.get_vote(a=agent, p=p) == -2:
                pass
            else:
                Na += 1
        return Na

    def active_Na(self):
        """
        Number of active agents where active agent is considered as one with Na greater than zero.
        :return: Number of agents which has Na > 0
        """
        count = 0
        for a in self.A_E():
            if self.agentNa(agent=a) > 0:
                count += 1
        return count

    def get_global_state(self) -> str:
        """
        Returns a string representing the global system state:
        paragraph_id | text | + total | - total
        """
        if not self.P():
            return ""
        lines = []
        header = f"{'Paragraph ID':<12} | {'Text':<40} | {'+ Votes':<7} | {'- Votes':<7}"
        separator = "-" * len(header)
        lines.append(header)
        lines.append(separator)

        for p in self.P():
            line = f"{p.name:<12} | {p.text[:40]:<40} | {self.p_plus(p):<7} | {self.p_minus(p):<7}"
            lines.append(line)

        return "\n".join(lines)

    def get_current_state_for_agent(self, agent: Agent) -> str:
        """
        Returns system state from the perspective of a specific agent:
        paragraph_id | text | + total | - total | own vote
        """
        if not self.P():
            return ""
        lines = []
        header = f"{'Paragraph ID':<12} | {'Text':<30} | {'+ Votes':<7} | {'- Votes':<7} | {'Own Vote':<9}"
        separator = "-" * len(header)
        lines.append(header)
        lines.append(separator)

        for p in self.P():
            own_vote = self.get_vote(agent, p)
            vote_display = (
                "+1" if own_vote == 1 else
                "-1" if own_vote == -1 else
                "0" if own_vote == 0 else
                "?"
            )
            line = f"{p.paragraph_id:<12} | {p.text[:30]:<30} | {self.p_plus(p):<7} | {self.p_minus(p):<7} | {vote_display:<9}"
            lines.append(line)

        return "\n".join(lines)

    def event_from_action(self, agent: 'Agent', action: Dict[str, str]) -> 'Event':
        """
        Convert a parsed agent action into an Event object using EventList state.
        Args:
            agent: The LLMAgent who made the decision.
            action: Parsed action dictionary.
        Example:
        {'action_id': 0,
         'type': 'VOTE',
         'paragraph_id': 2,
         'content': 'Subsidize electric vehicle purchases',
         'vote': 'UPVOTE',
         'reasoning': 'Supports balanced approach to reduce emissions while considering economic factors.'}
        :return: An event object.
        """
        paragraph_id = action["paragraph_id"]
        # Proposal event
        if action["type"] == "PROPOSE":
            paragraph = Paragraph(text=action["content"],
                                  paragraph_id=paragraph_id,
                                  name=f"p{paragraph_id}")
            vote = +1
        # Preference vote event
        elif action["type"] == "VOTE":
            paragraph = self.p_by_id(paragraph_id)
            vote = {"UPVOTE": +1, "DOWNVOTE": -1, "ABSTAIN": 0}.get(action["vote"], 0)
        else:
            raise ValueError("Invalid action type.")

        return Event(agent, paragraph, vote)

    def get_current_state_for_prompt(self, agent: Agent) -> str:
        """
        Returns system state from the perspective of a specific agent as a JSON-like string:
        Each paragraph includes full text, vote counts, and the agent's own vote.
        """
        if not self.P():
            return ""

        state_data = []
        for p in self.P():
            own_vote = self.get_vote(agent, p)
            vote_display = (
                "+1" if own_vote == 1 else
                "-1" if own_vote == -1 else
                "0" if own_vote == 0 else
                "?"
            )
            state_data.append({
                "paragraph_id": p.paragraph_id,
                "text": p.text,  # Full text, no truncation
                "votes_plus": self.p_plus(p),
                "votes_minus": self.p_minus(p),
                "own_vote": vote_display
            })

        # Return as a JSON-like string for LLM
        return "\n".join(str(p) for p in state_data)

    def get_current_state_solution_for_prompt(self, agent: Agent, solution: List[Paragraph]) -> str:
        """
        Returns system state from the perspective of a specific agent with the *solution* as a JSON-like string:
        Each paragraph includes full text, vote counts, and the agent's own vote.
        Additionally, it includes whether the paragraph is in the current solution.
        """
        # Empty event list
        if not self.P():
            return ""

        state_data = []
        for p in self.P():
            own_vote = self.get_vote(agent, p)
            include = "yes" if p in solution else "no"
            vote_display = (
                "+1" if own_vote == 1 else
                "-1" if own_vote == -1 else
                "0" if own_vote == 0 else
                "?"
            )
            state_data.append({
                "paragraph_id": p.paragraph_id,
                "text": p.text,  # Full text, no truncation
                "votes_plus": self.p_plus(p),
                "votes_minus": self.p_minus(p),
                "own_vote": vote_display,
                "In document": include
            })

        # Return as a JSON-like string for LLM
        return "\n".join(str(p) for p in state_data)
