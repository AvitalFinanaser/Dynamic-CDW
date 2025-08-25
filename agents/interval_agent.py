from agents.agent import Agent
from paragraphs import *
#from events import EventList
import random
import numpy as np
from abc import ABC, abstractmethod


# ~ Agent Model 2 - Euclidean


class EuclideanIntervalDistribution(ABC):
    """
    An abstract class for the Euclidean interval of the agent
    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        """
        Distribution of the center point
        """
        self.a = a
        self.b = b

    @abstractmethod
    def initializeInterval(self):
        """
        Initialize min and max scores using given distribution.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns the name of the distribution and its characteristics
        """
        pass


class UniformDistribution(EuclideanIntervalDistribution):
    def __init__(self, a: float = 0.0, b: float = 1.0):
        """
        Uniformal distribution of the center point
        """
        super().__init__(a=a, b=b)

    def initializeInterval(self):
        """
        Initialize min and max scores using uniform distribution.
        :return: A tuple (min_score, max_score) within [a, b]
        """
        middle_point = round(random.uniform(self.a, self.b), 2)
        radius = min(middle_point - self.a, self.b - middle_point)
        random_radius = np.random.uniform(0, radius)  # Random radius between 0 and max possible radius

        # Calculate min and max scores based on the middle point and radius
        min_score = round(max(self.a, middle_point - random_radius), 2)
        max_score = round(min(self.b, middle_point + random_radius), 2)
        # max_score = round(random.uniform(min_score, self.b), 2)
        return min_score, max_score

    def __str__(self) -> str:
        """
        Returns the name of the distribution and its characteristics
        """
        return f"Uniformal [{self.a},{self.b}]"


class GaussianDistribution(EuclideanIntervalDistribution):
    def __init__(self, a: float = 0.0, b: float = 1.0, mu: float = 0.5, sigma: float = 0.1):
        """
        Gaussian distribution of the center point
        """
        super().__init__(a=a, b=b)
        self.mu = mu
        self.sigma = sigma

    def initializeInterval(self):
        """
        Initialize min and max scores using a Gaussian (normal) distribution.
        The middle point is sampled from a normal distribution, and a random radius is added to determine the min/max scores.
        :return: A tuple (min_score, max_score) within [a, b]
        """
        middle_point = np.clip(np.random.normal(self.mu, self.sigma), self.a, self.b)

        # Sample a random radius ensuring the radius doesn't exceed the middle point's distance to the bounds
        radius = min(middle_point - self.a, self.b - middle_point)
        random_radius = np.random.uniform(0, radius)  # Random radius between 0 and max possible radius

        # Calculate min and max scores based on the middle point and radius
        min_score = round(max(self.a, middle_point - random_radius), 2)
        max_score = round(min(self.b, middle_point + random_radius), 2)

        return min_score, max_score

    def __str__(self) -> str:
        """
        Returns the name of the distribution and its characteristics
        """
        return f"Gaussian, mu:{self.mu}, sigma:{self.sigma} in range: [{self.a}, {self.b}]"


# Distibutions functions
def _initialize_uniform(min_score=None, max_score=None):
    """
    Initialize min and max scores using uniform distribution.
    :param min_score: Optional min score
    :param max_score: Optional max score
    :return: A tuple (min_score, max_score) within [0, 1]
    """
    if min_score is None:
        min_score = round(random.uniform(0, 1), 2)
    if max_score is None:
        max_score = round(random.uniform(min_score, 1), 2)
    return min_score, max_score


def _initialize_gaussian():
    """
    Initialize min and max scores using a Gaussian (normal) distribution.
    The middle point is sampled from a normal distribution, and a random radius is added to determine the min/max scores.
    :return: A tuple (min_score, max_score) within [0, 1]
    """
    mu = 0.5  # Mean for normal distribution (middle of the range)
    sigma = 0.1  # Standard deviation
    middle_point = np.clip(np.random.normal(mu, sigma), 0, 1)

    # Sample a random radius to generate min and max
    radius = np.clip(np.random.normal(0.1, 0.05), 0, middle_point)
    min_score = round(max(0, middle_point - radius), 2)
    max_score = round(min(1, middle_point + radius), 2)

    return min_score, max_score


def _initialize_gaussian_two_peak(agent_index, num_agents):
    """
    Initialize min and max scores using a two-peak Gaussian (normal) distribution.
    Half the agents have a distribution around 0.25, the other half around 0.75.
    :param agent_index: Index of the agent
    :param num_agents: Total number of agents
    :return: A tuple (min_score, max_score) within [0, 1]
    """
    if agent_index < num_agents / 2:
        # First half of agents - expectation around 0.25
        mu = 0.25
    else:
        # Second half of agents - expectation around 0.75
        mu = 0.75

    sigma = 0.05  # Standard deviation
    middle_point = np.clip(np.random.normal(mu, sigma), 0, 1)

    # Sample a random radius to generate min and max
    radius = np.clip(np.random.normal(0.1, 0.05), 0, middle_point)
    min_score = max(0, middle_point - radius)
    max_score = min(1, middle_point + radius)

    return min_score, max_score


class AgentInterval(Agent):
    """
    Represents an agent with a defined euclidian interval for paragraph scores
    Attributes:
        _min_score (float): The minimum score in the agent's interval
        _max_score (float): The maximum score in the agent's interval
        _distribution (str): The distribution type to the middle point ('uniform', 'gaussian', '2gaussian')
    """

    def __init__(self, agent_id: int, distribution: EuclideanIntervalDistribution = None, min_score=None,
                 max_score=None):
        """
        Initialize an AgentInterval instance.
        :param agent_id: The agent's unique identifier
        :param distribution: The type of distribution to sample the middle point
        :param min_score: The minimum score in the agent's interval
        :param max_score: The maximum score in the agent's interval
        """
        super().__init__(agent_id)

        # Case 1: Both min_score and max_score are provided
        if min_score is not None and max_score is not None:
            self._min_score = min_score
            self._max_score = max_score
            self._distribution = None  # No distribution needed when scores are manually set

        # Case 2: No min_score or max_score - use the distribution
        elif min_score is None and max_score is None:
            if distribution is None:
                self._distribution = UniformDistribution()  # Default to Uniform if no distribution is provided
            else:
                self._distribution = distribution
            self._min_score, self._max_score = self._distribution.initializeInterval()

        # Case 3: Partial data (either min_score or max_score is missing) - default to Uniform distribution
        else:
            raise ValueError("Both min_score and max_score must be provided together, or neither.")

    @property
    def agent_id(self):
        """
        Getter for agent_id.
        :return: The agent identifier
        """
        return self._agent_id

    @property
    def min_score(self):
        """
        Getter for min_score.
        :return: The minimum score in the agent's interval
        """
        return self._min_score

    @property
    def max_score(self):
        """
        Getter for max_score.
        :return: The maximum score in the agent's interval
        """
        return self._max_score

    @property
    def distribution(self):
        """
        Getter for distribution.
        :return: The distribution of the agent's interval
        """
        if self._distribution is None:
            return "None"
        return self._distribution

    def propose_paragraph(self, events):
        """
        Propose a new paragraph with a random score within the interval.
        :param events: The current event list the paragraph creates into
        :return: A new ParagraphWithScore instance with a score within interval values
        """
        text = f"Paragraph by agent {self.agent_id}"
        identifier = len(events.P()) + 1
        name = f"p{identifier}"
        score = random.uniform(self._min_score, self._max_score)
        return ParagraphWithScore(text=text, paragraph_id=identifier, name=name, score=score)

    # def propose_paragraph(available_paragraphs):
    #     """
    #     This method decides which new paragraph available (not suggested yet) to propose
    #     from the list of none suggested paragraphs.
    #     :param available_paragraphs: list of available paragraphs to suggest
    #     :return: A choice of paragraph to suggest
    #     """
    #     p = random.choice(available_paragraphs)  # Random choice of paragraph
    #     # p = f"Paragraph by agent {self.agent_id}"
    #     return p

    def vote_on_paragraph(self, paragraph):
        """
        Vote on an existing paragraph based on the agent's interval.
        :param paragraph: The paragraph to vote on
        :return: 1 if the paragraph's score is within the agent's interval, otherwise -1
        """
        if self._min_score <= paragraph.score <= self._max_score:
            return 1  # Vote in favor
        else:
            return -1  # Vote against

    def __str__(self):
        return f"Agent {self.agent_id} distribution is {self.distribution} with interval [{self.min_score}, {self.max_score}]"
