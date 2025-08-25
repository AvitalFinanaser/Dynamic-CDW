# tests/test_config.py
from simulation.config import SimulationConfig, AgentConfig, ContextConfig, AgentType
from agents import *
from simulation import *


def test_synthetic_config():
    """Shows how to create a config for synthetic paragraph generation"""
    # First, create the context configuration for synthetic paragraphs
    context_config = ContextConfig(
        context_type="synthetic",
        num_paragraphs=20
    )

    # Create a basic agent configuration - unstructured
    agent_config = AgentConfig(agent_type=AgentType.UNSTRUCTURED)

    # Create the full simulation configuration
    config = SimulationConfig(
        num_agents=10,
        num_events=100,
        num_lists=5,
        agent_config=agent_config,
        context_config=context_config
    )

    # Print out the configuration to verify
    print("\nSynthetic Configuration Test:")
    print(f"Number of agents: {config.num_agents}")
    print(f"Number of events: {config.num_events}")
    print(f"Number of lists: {config.num_lists}")
    print(f"Agent type: {config.agent_config.agent_type.value}")
    print(f"Context type: {config.context_config.context_type}")
    print(f"Number of paragraphs: {config.context_config.num_paragraphs}")


def test_dataset_config():
    """Shows how to create a config for dataset-based paragraph generation"""
    # Create context configuration for dataset
    context_config = ContextConfig(
        context_type="dataset",
        dataset_path="data/paragraphs.csv"
    )

    # Create agent configuration with distribution
    agent_config = AgentConfig(
        agent_type=AgentType.INTERVAL,
        distribution=UniformDistribution(),
    )

    # Create full configuration
    config = SimulationConfig(
        num_agents=5,
        num_events=50,
        num_lists=3,
        random_seed=123,  # Custom random seed
        agent_config=agent_config,
        context_config=context_config
    )

    # Print out the configuration to verify
    print("\nDataset Configuration Test:")
    print(f"Number of agents: {config.num_agents}")
    print(f"Random seed: {config.random_seed}")
    print(f"Context type: {config.context_config.context_type}")
    print(f"Dataset path: {config.context_config.dataset_path}")
    print(f"Agent type value: {config.agent_config.agent_type.value}")
    print(f"Agent distribution: {config.agent_config.distribution}")
    print(f"Proportion 25: {config.agent_config.proportion_25}")


def test_configuration_reproducibility():
    """
    Tests whether the same configuration produces identical event lists
    when using the same seed value.
    """
    # Create a test configuration with a specific seed
    config = SimulationConfig(
        num_agents=50,
        num_events=20,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=50),
    )

    # Create two schedulers with the same configuration
    scheduler1 = Scheduler(config)
    scheduler2 = Scheduler(config)

    # Generate event lists from both schedulers
    community1, events1 = scheduler1.schedule_single_instance()
    community2, events2 = scheduler2.schedule_single_instance()

    # Compare the event lists
    print("Testing event list reproducibility:")
    print("Are event lists identical?", events1.equals(events2))
    print("\nFirst few events from list 1:")
    print(events1.events_df().head())
    print("\nFirst few events from list 2:")
    print(events2.events_df().head())


if __name__ == "__main__":
    # Run both tests and see their output
    #test_synthetic_config()
    #test_dataset_config()
    test_configuration_reproducibility()

