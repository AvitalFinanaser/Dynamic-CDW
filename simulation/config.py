# This module will decide on the configuration parameters of the system

from dataclasses import dataclass
from enum import Enum
from agents import *


@dataclass
class ContextConfig:
    # The type can be either "dataset" or "synthetic"
    context_type: str = None
    # For dataset option, we need the file path
    dataset_path: str = None
    # For synthetic option, we need number of paragraphs
    num_paragraphs: int = None
    # For synthetic option, we need number of paragraphs
    topic: str = "Climate change policy"


class AgentType(Enum):
    """Defines the available types of agents in our simulations"""
    UNSTRUCTURED = "unstructured"
    INTERVAL = "euclidean_intervals"
    TEXTUAL = "llm"


@dataclass
class AgentConfig:
    """Configuration specific to agent properties"""
    agent_type: AgentType
    distribution: EuclideanIntervalDistribution = None
    proportion_25: float = 0


@dataclass
class SimulationConfig:
    """Main configuration class that holds all simulation parameters"""
    # Basic simulation parameters
    num_agents: int
    num_events: int
    num_lists: int

    # Nested configurations
    agent_config: AgentConfig
    context_config: ContextConfig

    # Fields with default values
    random_seed: int = 42
