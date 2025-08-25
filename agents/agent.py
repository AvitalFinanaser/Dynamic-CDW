import random
from typing import List


class Agent:

    def __init__(self, agent_id):
        """
        agent_id (integer): unique identifier for each agent
        name (string): name of the agent
        gender (string): 'M' for male and 'F' for female
        age (integer): the age of the agent
        description (string): small description of the agent
        """
        self.agent_id = agent_id  # Agent's identifier
        self.name = f"a{agent_id}"

    def __str__(self):
        return f"Agent: a{self.agent_id}, named: {self.name}"
