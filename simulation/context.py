# simulation/context.py
from dataclasses import dataclass
from typing import Type, Optional
from agents import *


@dataclass
class SimulationContext:
    " Maybe it can has only the relataion to P and A parametes "
    # Basic simulation parameters
    agent_type: Type
    num_agents: int
    num_events: int
    num_lists: int

    # Additional configuration parameters
    distribution: EuclideanIntervalDistribution = None
    proportion_25: float = 0
    random_seed: int = 42
    # Add any other configuration parameters you need