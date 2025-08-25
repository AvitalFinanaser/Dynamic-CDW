import multiprocessing
import os
import json
from datetime import datetime
from simulation import *
from agents import *
from rules import *


def generate_experiment_rules():
    """
    Generates rules for CDW system experiments.
    1. static - CSF: ["APS"]
    2. parma (dynamic property) - events, paragraphs
    3. alpha (smoothing parameter) - (0.05,0.1,0.3,0.5)
    """
    # Rules options
    threshold = 0.5
    CSF = ["APS"]
    d_events = EventsDynamicProperty()
    d_paragraphs = ParagraphsDynamicProperty()
    Fexp_05 = ExpSmoothingFunction(alpha=0.05)
    Fexp_1 = ExpSmoothingFunction(alpha=0.1)
    Fexp_3 = ExpSmoothingFunction(alpha=0.3)
    Fexp_5 = ExpSmoothingFunction(alpha=0.5)
    rules = []

    # 1-SC: CSF: {APS, "AM", "APS_r"}
    cS1 = StaticCondition(CSF="APS", threshold=threshold)
    cS2 = StaticCondition(CSF="APS_r", threshold=threshold)
    cS3 = StaticCondition(CSF="AM", threshold=threshold, beta=0.05)
    cS4 = StaticCondition(CSF="AM", threshold=threshold, beta=0.1)

    rules.extend([cS1, cS2, cS3, cS4])

    # harshD: CSF: {"APS", "AM", "APS_r"}, property:{events, paragraphs}, t = [0, 50, 100]
    hD1 = HarshDynamicCondition(CSF="APS", threshold=threshold, dynamic_property=d_events, t=50)
    hD2 = HarshDynamicCondition(CSF="APS", threshold=threshold, dynamic_property=d_paragraphs, t=50)

    hD3 = HarshDynamicCondition(CSF="APS_r", threshold=threshold, dynamic_property=d_events, t=50)
    hD4 = HarshDynamicCondition(CSF="APS_r", threshold=threshold, dynamic_property=d_paragraphs, t=50)

    hD5 = HarshDynamicCondition(CSF="AM", threshold=threshold, dynamic_property=d_events, t=50, beta=0.05)
    hD6 = HarshDynamicCondition(CSF="AM", threshold=threshold, dynamic_property=d_paragraphs, t=50, beta=0.05)

    hD7 = HarshDynamicCondition(CSF="AM", threshold=threshold, dynamic_property=d_events, t=50, beta=0.1)
    hD8 = HarshDynamicCondition(CSF="AM", threshold=threshold, dynamic_property=d_paragraphs, t=50, beta=0.1)

    hD9 = HarshDynamicCondition(CSF="APS", threshold=threshold, dynamic_property=d_events, t=0)

    rules.extend([hD1, hD2, hD3, hD4, hD5, hD6, hD7, hD8, hD9])

    return rules


def generate_experiment_configurations():
    """
    Generates configurations for CDW system experiments with ordered IDs that match
    the logical grouping of parameters. Configurations are ordered by:
    1. Population size (ascending)
    2. Event count (ascending within each population)
    3. Agent type (UNSTRUCTURED first, then INTERVAL with different distributions)
    """
    # Define core parameters with ordered lists
    populations = sorted([20])  # Ensure populations are ordered
    # Sort event counts for each population
    events_map = {
        # 10: sorted([100, 200, 300]),
        20: sorted([200, 300, 400]),
        # 50: sorted([300, 400, 500])
    }
    num_instances = 20

    # Create ordered list of agent configurations
    agent_configs = [
        # 1. Unstructured
        {
            'config': AgentConfig(agent_type=AgentType.UNSTRUCTURED),
            'order_key': 'A_UNSTRUCTURED'
        },
        # 2. Interval with Uniform
        {
            'config': AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=UniformDistribution()
            ),
            'order_key': 'B_INTERVAL_UNIFORM'
        },
        # 3. Interval with Single Gaussian
        {
            'config': AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=GaussianDistribution(mu=0.5, sigma=0.1)
            ),
            'order_key': 'C_INTERVAL_GAUSSIAN'
        },
        # 4. Interval with Two-Peak Gaussian
        {
            'config': AgentConfig(
                agent_type=AgentType.INTERVAL,
                distribution=GaussianDistribution(mu=0.5, sigma=0.1),
                proportion_25=0.3
            ),
            'order_key': 'D_INTERVAL_TWOPEAK'
        }
    ]

    # Generate configurations in order
    configurations = []
    config_counter = 1

    # Iterate in the desired order
    for population in populations:
        for events in events_map[population]:
            for agent_cfg in agent_configs:
                config = SimulationConfig(
                    num_agents=population,
                    num_events=events,
                    num_lists=num_instances,
                    agent_config=agent_cfg['config'],
                    context_config=ContextConfig(
                        context_type="synthetic",
                        num_paragraphs=population
                    )
                )

                configurations.append({
                    'id': f'config{config_counter:03d}',  # Pad with zeros for consistent sorting
                    'config': config,
                    'order_key': f"{population:03d}_{events:03d}_{agent_cfg['order_key']}"
                })
                config_counter += 1

    # Sort configurations by our composite order key
    configurations.sort(key=lambda x: x['order_key'])

    # Remove the order key as it's no longer needed
    for config in configurations:
        del config['order_key']

    return configurations


def process_single_configuration(args):
    """
    Processes all rules for a single configuration on one CPU core.
    This function contains all the logic needed to analyze one configuration
    with all rules and save the results.

    Parameters:
    -----------
    args : tuple
        Contains (config_dict, rules, base_dir) for processing
    """
    config_dict, rules, base_dir = args
    config_id = config_dict['id']
    config = config_dict['config']

    print(f"\nProcessing {config_id}")
    print(f"Population: {config.num_agents}, Events: {config.num_events}, Agents: {config.agent_config.agent_type}")

    # Initialize system for this configuration
    scheduler = Scheduler(config)
    system = CDWSystem(scheduler)

    # Create directory for this configuration
    config_dir = os.path.join(base_dir, config_id)
    os.makedirs(config_dir, exist_ok=True)

    # Test each rule
    for rule in rules:
        # Create clean rule identifier for directory naming
        rule_id = f"{rule.__str__.replace(' ', '_')}"
        rule_dir = os.path.join(config_dir, rule_id)
        os.makedirs(rule_dir, exist_ok=True)

        try:
            # Generate combined metrics for this configuration-rule pair
            results = system.analyze_multiple_metrics(rule)

            # Save the metrics data
            metrics_path = os.path.join(rule_dir, f"{rule_id}.csv")
            results.to_csv(metrics_path, index=False)

            # Save metadata about this analysis run
            metadata = {
                'configuration': {
                    'id': config_id,
                    'population': config.num_agents,
                    'events': config.num_events,
                    'agent_type': config.agent_config.agent_type.value,
                    'num_instances': config.num_lists
                },
                'rule': rule.__str__,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

            metadata_path = os.path.join(rule_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"Saved analysis for {config_id} with {rule_id}")

        except Exception as e:
            print(f"Error processing {config_id} with {rule_id}: {str(e)}")
            continue

    return f"Completed configuration {config_id}"


def run_parallel_analysis(configs, rules, base_dir="results/parallel", num_processes=11):
    """
    Runs the analysis using multiple CPU cores.
    Distributes configurations across available cores and processes all rules
    for each configuration.

    Parameters:
    -----------
    configs : list
        List of configuration dictionaries
    rules : list
        List of rule objects to test
    base_dir : str
        Base directory for saving results
    num_processes : int
        Number of CPU cores to use (default: 11, leaving one core free)
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Prepare arguments for each configuration
    processing_args = [(config, rules, base_dir) for config in configs]

    print(f"\nStarting parallel processing with {num_processes} cores")
    print(f"Total configurations to process: {len(configs)}")
    print(f"Rules per configuration: {len(rules)}")

    # Create and run the process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_configuration, processing_args)

    print("\nAll configurations processed!")
    for result in results:
        print(result)


def generate_experiment_dynamic_rules():
    """
    Generates rules for CDW system experiments with expanded CSF functions
    """
    # CSF
    threshold = 0.5
    CSF_types = ["APS", "APS_r", "AM"]
    # Dynamic properties
    d_events = EventsDynamicProperty()
    d_paragraphs = ParagraphsDynamicProperty()
    dynamic_properties = [d_events, d_paragraphs]
    # Smoothing parameters
    alphas = [0.05, 0.1, 0.3, 0.5, 0.8, 1]
    smoothing_functions = {
        f"Fexp_{str(alpha).replace('.', '')}": ExpSmoothingFunction(alpha=alpha)
        for alpha in alphas
    }
    betas = [0.05, 0.1, 0.3, 0.5]

    rules = []

    # Smooth Dynamic Conditions
    for csf in CSF_types:
        for prop in dynamic_properties:
            for f_name, f in smoothing_functions.items():
                # Beta parameter must be defined
                if csf == "AM":
                    for beta in betas:
                        rules.append(
                            SmoothDynamicCondition(
                                CSF=csf,
                                threshold=threshold,
                                dynamic_property=prop,
                                F=f,
                                beta=beta
                            )
                        )
                else:
                    rules.append(
                        SmoothDynamicCondition(
                            CSF=csf,
                            threshold=threshold,
                            dynamic_property=prop,
                            F=f
                        )
                    )

    return rules


if __name__ == "__main__":
    # Generate configurations and rules
    rules = generate_experiment_dynamic_rules()
    for rule in rules:
        print(rule)
    configs = generate_experiment_configurations()
    for config in configs:
        config_id = config['id']
        config = config['config']
        print(f"\nConfig: {config_id}")
        print(f"Population: {config.num_agents}, Events: {config.num_events}, Agents: {config.agent_config.agent_type}, Dist: {config.agent_config.distribution}")

    #rules = generate_experiment_rules()

    # Run the parallel analysis
    #run_parallel_analysis(configs=configs, rules=rules)
