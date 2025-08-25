from simulation.scheduler import *
from simulation.cdw_system import CDWSystem
from rules import *
import os
import pandas as pd


def test_system_instance_data():
    """
    Tests retrieving instance data from the system by creating a simple configuration
    and printing the results.
    """
    config = SimulationConfig(
        num_agents=20,
        num_events=200,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=200)
    )
    # Initialize system
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)

    # Test single instance
    print("\nTesting single instance retrieval:")
    agents, event_list = system.get_single_instances_data()
    print(f"Number of agents: {len(agents)}")
    print(f"Number of events: {len(event_list.events)}")
    print(f"Event list: {event_list.events_df()}")

    # Test multiple instances
    print("\nTesting multiple instances retrieval:")
    instances = system.get_multiple_instances_data()
    print(f"Number of instances: {len(instances)}")
    for i, (agents, events) in enumerate(instances, 1):
        print(f"Instance {i}:")
        print(f"  Agents: {len(agents)}")
        print(f"  Events: {len(events.events)}")


def test_system_satisfaction():
    """
    Tests the satisfaction analysis by creating a system with a simple configuration
    and printing the analysis results.
    """
    config = SimulationConfig(
        num_agents=20,
        num_events=10,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=20)
    )
    # Initialize system
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)
    rule = StaticCondition(CSF="APS", threshold=0.5)

    # Run satisfaction analysis
    print("\nTesting single instance satisfaction analysis:")

    print(system.analyze_single_instance_satisfaction(rule))

    print("\nTesting multiple satisfaction analysis:")
    results = system.analyze_multiple_satisfaction(rule)
    print(results)


def test_system_stability():
    """
    Tests the stability analysis by creating a system with a simple configuration
    and printing the analysis results.
    """
    # Create configuration
    config = SimulationConfig(
        num_agents=20,
        num_events=200,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=20)
    )

    # Initialize system and rule
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)
    rule = StaticCondition(CSF="AM", threshold=0.5, beta=0.05)

    # Run stability analysis
    print("\nTesting stability analysis:")
    results = system.analyze_multiple_stability(rule)
    print(results)
    return(results)


def test_system_combined_analysis():
    """
    Tests the combined analysis of satisfaction and stability metrics.
    Creates a system with a specific configuration, runs the analysis,
    and displays the comprehensive results.
    """
    print("\nTesting Combined System Analysis")
    print("================================")

    # Create a test configuration with meaningful parameters
    config = SimulationConfig(
        num_agents=20,
        num_events=200,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=20)
    )

    # Initialize system and rule
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)
    rule = StaticCondition(CSF="APS", threshold=0.5)
    rule = StaticCondition(CSF="AM", threshold=0.5, beta=0.1)

    # Run combined analysis
    print(f"\nRunning combined satisfaction and stability analysis:\n Rule: {rule}")
    results = system.analyze_multiple_metrics(rule)

    # Display results in a formatted way
    print("\nCombined Analysis Results:")
    print("\nFirst few rows of the combined metrics:")
    print(results)

    # Show correlation between metrics

    # print("\nCorrelations between Satisfaction and Stability:")
    # correlations = results.corr()
    # print(correlations)
    #
    # print("\nTest completed!")


def test_system_analysis_plots():
    """
    Tests the system's analysis visualization methods with automatic file organization.
    """
    # Create test configuration
    config = SimulationConfig(
        num_agents=20,
        num_events=200,
        num_lists=5,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=20)
    )

    # Initialize system and rule
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)
    rule = StaticCondition(CSF="APS", threshold=0.5)

    # Get analysis results
    analysis_results = system.analyze_multiple_metrics(rule)

    # Generate plots
    system.plot_stability_analysis(rule=rule, analysis_results=analysis_results)
    system.plot_satisfaction_analysis(rule=rule, analysis_results=analysis_results)


# Run all tests
if __name__ == "__main__":
    """
    Runs all system tests sequentially, displaying results from each test.
    """
    print("Testing CDWSystem Functionality")
    print("==============================")
    # test_system_instance_data()
    # test_system_satisfaction()
    # results = test_system_stability()
    test_system_combined_analysis()

    # Saving the analysis into the project folder
    # df = test_system_analysis_plots()
    # save_directory = 'CDW/results/analysis'
    # os.makedirs(save_directory, exist_ok=True)
    # file_path = os.path.join(save_directory, 'test_unite.csv')
    # df.to_csv(file_path, index=False)
    # print(f"CSV file saved to {file_path}")
    # print("\nAll tests completed!")
