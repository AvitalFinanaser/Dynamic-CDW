import multiprocessing
from datetime import datetime

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List

from simulation import *
from events import *
from agents import *
from paragraphs import *
from agents import *
from rules import *
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# ---------- PART 1: Saving Set up -------------


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

    save_root = Path("datasets/event_lists")
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


def save_communities(config_dict, communities):
    base_path = Path("datasets/communities") / config_dict['id']
    base_path.mkdir(parents=True, exist_ok=True)
    for idx, agents in enumerate(communities):
        agents_data = [
            {
                "agent_id": agent.agent_id,
                "profile": agent.profile,
                "topic": agent.topic,
                "topic_position": agent.topic_position
            } for agent in agents
        ]
        with open(base_path / f"community_{idx}.json", "w") as f:
            json.dump(agents_data, f, indent=2)


def load_communities(config_dict):
    """
    Loads previously saved communities for a given configuration.

    Args:
        config_dict: Dictionary containing 'id' and 'config' keys.

    Returns:
        List of communities, where each community is a list of LLMAgent objects.
    """
    base_path = Path("datasets/communities") / config_dict['id']
    communities = []

    for instance_id in range(config_dict['config'].num_lists):
        file_path = base_path / f"community_{instance_id}.json"
        with open(file_path, "r") as f:
            agents_data = json.load(f)

        community = [
            LLMAgent(
                agent_id=agent["agent_id"],
                profile=agent["profile"],
                topic=agent["topic"],
                topic_position=agent["topic_position"]
            )
            for agent in agents_data
        ]
        communities.append(community)

    return communities


# ---------- PART 2: SETUP AND CONFIGURATION -------------


def generate_experiment_rules():
    """
    Generates rules for CDW system experiments with expanded CSF functions.
    """
    threshold = 0.5
    CSF_types = ["APS", "APS_r", "AM"]
    d_events = EventsDynamicProperty()
    d_paragraphs = ParagraphsDynamicProperty()
    dynamic_properties = [d_events, d_paragraphs]
    alphas = [0.1, 0.3, 0.5, 1]
    smoothing_functions = {
        f"Fexp_{str(alpha).replace('.', '')}": ExpSmoothingFunction(alpha=alpha)
        for alpha in alphas
    }
    t_values = [0, 100, 150]
    betas = [0.05, 0.1, 0.3, 0.5]

    rules = []

    # Static rules
    for csf in CSF_types:
        # For each Dynamic property
        if csf == "AM":
            # AM requires beta
            for beta in [0.05, 0.1, 0.3]:
                rules.append(
                    StaticCondition(
                        CSF=csf,
                        threshold=threshold,
                        beta=beta
                    )
                )
        else:
            rules.append(
                StaticCondition(
                    CSF=csf,
                    threshold=threshold,
                )
            )

    # Harsh rules
    for csf in ["APS", "APS_r"]:
        for prop in dynamic_properties:
            # For each t vale
            for t in t_values:
                # For each Dynamic property
                rules.append(
                    HarshDynamicCondition(
                        CSF=csf,
                        threshold=threshold,
                        dynamic_property=prop,
                        t=t,
                    ))

    # Smooth rules

    # For each CSF
    for csf in CSF_types:
        # For each Dynamic property
        for prop in dynamic_properties:
            # For each smoothing function
            for f_name, f in smoothing_functions.items():
                if csf == "AM":
                    # AM requires beta
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


def generate_experiment_configurations():
    """
    Generates configurations for CDW system experiments with ordered IDs that match
    the logical grouping of parameters.
    """
    # You can expand population sizes & events_map for more tasks
    populations = sorted([20, 40])
    events_map = {
        20: sorted([150, 250]),
        40: sorted([50]),
    }
    num_instances = 5

    # Define agent configurations
    agent_configs = [
        # # 1. Unstructured
        # {
        #     'config': AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        #     'order_key': 'A_UNSTRUCTURED'
        # },

        # # 2. Interval with Uniform
        # {
        #     'config': AgentConfig(
        #         agent_type=AgentType.INTERVAL,
        #         distribution=UniformDistribution()
        #     ),
        #     'order_key': 'B_INTERVAL_UNIFORM'
        # }
        # ,

        # 3. Interval with Single Gaussian
        # {
        #     'config': AgentConfig(
        #         agent_type=AgentType.INTERVAL,
        #         distribution=GaussianDistribution(mu=0.5, sigma=0.1)
        #     ),
        #     'order_key': 'C_INTERVAL_GAUSSIAN'
        # },

        # # 4. Interval with Two-Peak Gaussian
        # {
        #     'config': AgentConfig(
        #         agent_type=AgentType.INTERVAL,
        #         distribution=GaussianDistribution(mu=0.5, sigma=0.1),
        #         proportion_25=0.3
        #     ),
        #     'order_key': 'D_INTERVAL_TWOPEAK'
        # }

        # 5. LLM agent - Textual
        {
            'config': AgentConfig(
                agent_type=AgentType.TEXTUAL,
            ),
            'order_key': 'E_Textual'
        }
    ]

    configurations = []
    config_counter = 3

    for population in populations:
        for events in events_map[population]:
            for agent_cfg in agent_configs:
                config_obj = SimulationConfig(
                    num_agents=population,
                    num_events=events,
                    num_lists=num_instances,
                    agent_config=agent_cfg['config'],
                    context_config=ContextConfig(
                        context_type="synthetic",
                        num_paragraphs=population
                    ),
                    random_seed=52
                )
                configurations.append({
                    'id': f'config{config_counter:03d}_{config_obj.agent_config.agent_type.value}',
                    'config': config_obj,
                    'order_key': f"{population:03d}_{events:03d}_{agent_cfg['order_key']}"
                })
                config_counter += 1

    # Sort by 'order_key' for consistent ordering
    configurations.sort(key=lambda x: x['order_key'])

    # Remove the key after sorting
    for cfg_dict in configurations:
        del cfg_dict['order_key']

    return configurations


# ---------- PART 3: EVENT LIST SCHEDULING (WORKER) -------------

def worker_schedule_events(args):
    config_id, config, instance_id, rule = args

    worker_name = multiprocessing.current_process().name
    start_time = datetime.now()
    print(
        f"{worker_name} [START SCHEDULING at {start_time.strftime('%Y-%m-%d %H:%M:%S')}] => Config: {config_id}, Instance {instance_id}, Rule: {rule}",
        flush=True)

    try:
        community_path = Path("datasets/communities") / config_id / f"community_{instance_id}.json"

        with open(community_path, "r") as f:
            agents_data = json.load(f)

        agents = [LLMAgent(**data) for data in agents_data]

        event_list = Scheduler.schedule_eventlist_textual_Rule(
            agents=agents,
            num_events=config.num_events,
            rule=rule
        )

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

        save_dir = Path(
            "datasets/event_lists") / config_id / f"{str(rule).replace(' ', '_')}" / f"instance_{instance_id}"

        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "agents.json", "w") as f:
            json.dump(agents_data, f, indent=2)
        with open(save_dir / "events.json", "w") as f:
            json.dump(events_data, f, indent=2)
        with open(save_dir / "paragraphs.json", "w") as f:
            json.dump(paragraphs_data, f, indent=2)

        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = datetime.now() - start_time

        return f"{worker_name} [END SCHEDULING at {end_time}, took {duration}] => Config: {config_id}, Instance {instance_id}, Rule: {rule}"

    except Exception as e:
        err_msg = f"{worker_name} [ERROR] Config: {config_id}, Instance: {instance_id}, Rule: {rule}, Error: {e}"
        print(err_msg, flush=True)
        return err_msg


# ---------- PART 4: FULL PARALLEL SCHEDULING -------------


def parallel_schedule(configs, rules):
    tasks = []
    for config_dict in configs:
        for instance_id in range(config_dict['config'].num_lists):
            for rule in rules:
                # Each worker works on scheduling event list from (config, instance, rule)
                tasks.append((config_dict['id'], config_dict['config'], instance_id, rule))

    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        for result in pool.imap_unordered(worker_schedule_events, tasks):
            print(result)


# ---------- PART 5: METRIC ANALYSIS (WORKER) -------------

def worker_analyze_metrics(args):
    config_dict, rule = args
    config_id = config_dict['id']
    config = config_dict['config']

    # Print start time
    worker_name = multiprocessing.current_process().name
    start_time = datetime.now()
    print(
        f"{worker_name} [START METRICS at {start_time.strftime('%Y-%m-%d %H:%M:%S')}]  => Config: {config_id}, Rule: {rule}",
        flush=True)

    try:
        # Load ALL instances for given config_id and rule
        base_path = Path("datasets/event_lists") / config_id / f"{str(rule).replace(' ', '_')}"
        instance_dirs = sorted(base_path.glob("instance_*"))

        # Load communities behind scheduling
        communities = load_communities(config_dict)

        # Retrieve all instances
        instances_data = []
        for instance_id, instance_dir in enumerate(instance_dirs):
            event_list = load_event_list(instance_dir)
            agents = communities[instance_id]
            instances_data.append((agents, event_list))

        # Calculate metrics
        dummy_config = SimulationConfig(
            num_agents=0,
            num_events=0,
            num_lists=0,
            agent_config=AgentConfig(agent_type=AgentType.TEXTUAL),
            context_config=ContextConfig(context_type="synthetic", num_paragraphs=0),
            random_seed=0
        )
        system = CDWSystem(Scheduler(dummy_config))
        results = system.analyze_multiple_metrics_together(rule, instances_data)

        # Results saving
        results_dir = Path("datasets/metrics")
        results_dir.mkdir(parents=True, exist_ok=True)
        # Create nested directory structure: metrics/config_id/rule_name/
        config_dir = results_dir / config_id
        config_dir.mkdir(exist_ok=True)
        rule_dir = config_dir / str(rule).replace(' ', '_')
        rule_dir.mkdir(exist_ok=True)

        filename = f"{config_id}_{str(rule).replace(' ', '_')}.csv"
        results.to_csv(rule_dir / filename, index=False)

        # Metadata
        metadata = {
            'configuration': {
                'id': config_id,
                'population': config.num_agents,
                'events': config.num_events,
                'agent_type': config.agent_config.agent_type.value,
                'num_instances': config.num_lists
            },
            'rule': str(rule),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        metadata_filename = f"{config_id}_{str(rule).replace(' ', '_')}_metadata.json"
        with open(rule_dir / metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Print ending message
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        time_difference = datetime.now() - start_time
        print(f"{worker_name} [END METRICS at {end_time}, took {time_difference}] => Config: {config_id}, Rule: {rule}",
              flush=True)

        return f"{worker_name} [COMPLETED METRICS] => Config: {config_id}, Rule: {rule}"

    except Exception as e:
        err_msg = f"{worker_name} [ERROR - METRICS] Config: {config_id}, Rule: {rule}, Error: {e}"
        print(err_msg, flush=True)
        return err_msg


# ---------- PART 6: FULL PARALLEL ANALYSIS ---------------

def parallel_analysis(configs, rules):
    tasks = []
    for config_dict in configs:
        for rule in rules:
            tasks.append((config_dict, rule))
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
        for result in pool.imap_unordered(worker_analyze_metrics, tasks):
            print(result)


def process_config_rule_pair(args):
    """
    Process a single (config, rule) pair in a dedicated worker process.
    """
    config_dict, rule, base_dir = args
    config_id = config_dict['id']
    config = config_dict['config']

    # Get the name of this worker process (e.g. "SpawnPoolWorker-1", etc.)
    worker_name = multiprocessing.current_process().name

    # Print a "start" message with timestamp, rule, config, etc.
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[START {start_time}] {worker_name} => Config: {config_id}, Rule: {rule}", flush=True)

    try:
        # Set up the system
        scheduler = Scheduler(config)
        system = CDWSystem(scheduler)
        agents = scheduler.create_community()
        event_list = Scheduler.schedule_eventlist_textual_Rule(
            agents=agents,
            num_events=config.num_events,
            rule=rule
        )

        # Prepare directories
        config_dir = os.path.join(base_dir, config_id)
        os.makedirs(config_dir, exist_ok=True)

        rule_id = f"{str(rule).replace(' ', '_')}"
        rule_dir = os.path.join(config_dir, rule_id)
        os.makedirs(rule_dir, exist_ok=True)

        # Run analysis
        results = system.analyze_multiple_metrics(rule)

        # Save CSV
        csv_path = os.path.join(rule_dir, f"{rule_id}.csv")
        results.to_csv(csv_path, index=False)

        # Save metadata
        metadata = {
            'configuration': {
                'id': config_id,
                'population': config.num_agents,
                'events': config.num_events,
                'agent_type': config.agent_config.agent_type.value,
                'agent_dist': config.agent_config.distribution.value,
                'agent_dist_proportion_25': config.agent_config.proportion_25.value,
                'num_instances': config.num_lists
            },
            'rule': str(rule),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        meta_path = os.path.join(rule_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Print a "done" message
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[DONE {end_time}] {worker_name} => Config: {config_id}, Rule: {rule_id}", flush=True)
        return f"SUCCESS: {config_id}, {rule_id}"

    except Exception as e:
        err_msg = f"ERROR in {config_id}, rule {rule}: {str(e)}"
        print(err_msg, flush=True)
        return err_msg


def run_parallel_analysis(configs, rules, base_dir="results/parallel/llm_agents"):
    """
    Parallelize at the (config, rule) level to maximize CPU usage.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Build a list of all tasks: (config_dict, rule, base_dir)
    tasks = []
    for config_dict in configs:
        for rule in rules:
            tasks.append((config_dict, rule, base_dir))

    # Number of CPU cores to use: all but one by default
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"\nUsing {num_cores} worker processes.")
    print(f"Total configurations: {len(configs)}")
    print(f"Total rules: {len(rules)}")
    print(f"Total tasks (config-rule pairs): {len(tasks)}\n")

    # Use a Pool and dispatch tasks
    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered returns results as they complete
        for result in pool.imap_unordered(process_config_rule_pair, tasks, chunksize=1):
            # Print or log each result
            pass

    print("\nAll tasks completed.")


# ---------- PART 7: TESTING ------------------------------

def test_worker_schedule():
    configs = generate_experiment_configurations()
    rules = generate_experiment_rules()

    # Just test one config, one instance, one rule quickly
    config_dict = configs[0]
    scheduler = Scheduler(config_dict['config'])
    communities = scheduler.create_communities()
    save_communities(config_dict, communities)

    args = (config_dict['id'], config_dict['config'], 0, rules[0])
    result = worker_schedule_events(args)
    print(result)


def test_worker_analysis():
    """
    Test the worker_analyze_metrics function with the new analyze_multiple_metrics_together method.
    This tests the calculation of metrics for a single configuration and rule.
    """
    configs = generate_experiment_configurations()
    rules = generate_experiment_rules()

    # Get the first config and rule for testing
    config_dict = configs[0]
    rule = rules[0]

    # Print test info
    print(f"\nTESTING ANALYSIS for:")
    print(f"  Config: {config_dict['id']}")
    print(f"  Rule: {rule}")

    try:
        # Load communities (assuming they've been created)
        communities = load_communities(config_dict)

        # Create a base path for the test event list
        base_path = Path("datasets/event_lists") / config_dict['id'] / f"{str(rule).replace(' ', '_')}"
        if not base_path.exists():
            print(f"Creating test event list path: {base_path}")
            base_path.mkdir(parents=True, exist_ok=True)

        # Load instances
        instance_dirs = sorted(base_path.glob("instance_*"))
        print(f"Found {len(instance_dirs)} instance(s) for testing")

        # Load the data for analysis
        instances_data = []
        for instance_id, instance_dir in enumerate(instance_dirs):
            event_list = load_event_list(instance_dir)
            agents = communities[instance_id]
            instances_data.append((agents, event_list))

        # Create system for analysis
        dummy_config = SimulationConfig(
            num_agents=0,
            num_events=0,
            num_lists=0,
            agent_config=AgentConfig(agent_type=AgentType.TEXTUAL),
            context_config=ContextConfig(context_type="synthetic", num_paragraphs=0),
            random_seed=0
        )
        system = CDWSystem(Scheduler(dummy_config))

        # Test the original method
        print("\nTesting original analyze_multiple_metrics method...")
        start_time = datetime.now()
        original_results = system.analyze_multiple_metrics(rule, instances_data)
        original_time = datetime.now() - start_time
        print(f"Original method took {original_time}")

        # Test the new optimized method
        print("\nTesting new analyze_multiple_metrics_together method...")
        start_time = datetime.now()
        new_results = system.analyze_multiple_metrics_together(rule, instances_data)
        new_time = datetime.now() - start_time
        print(f"New method took {new_time}")

        # Compare results
        print("\nComparing results:")

        # Check if the dataframes have the same columns
        original_cols = set(original_results.columns)
        new_cols = set(new_results.columns)

        missing_in_new = original_cols - new_cols
        extra_in_new = new_cols - original_cols

        if missing_in_new:
            print(f"Warning: New method is missing columns: {missing_in_new}")
        if extra_in_new:
            print(f"Note: New method has additional columns: {extra_in_new}")

        # Compare values for common columns
        common_cols = original_cols.intersection(new_cols)
        print(f"Comparing {len(common_cols)} common columns...")

        # Check event counts
        if len(original_results) != len(new_results):
            print(f"Warning: Different number of events: Original={len(original_results)}, New={len(new_results)}")

        match_count = 0
        diff_count = 0

        for col in common_cols:
            if col == 'Event':  # Skip the event column
                continue

            # Check if values are approximately equal (floating point comparison)
            if original_results[col].equals(new_results[col]):
                match_count += 1
            else:
                # Calculate max difference
                max_diff = 0
                try:
                    max_diff = abs(original_results[col] - new_results[col]).max()
                except:
                    max_diff = "N/A (non-numeric)"

                print(f"  Column '{col}' differs, max difference: {max_diff}")
                diff_count += 1

        print(f"\nResults: {match_count} columns match, {diff_count} columns differ")

        # Time comparison
        speedup = original_time / new_time if new_time.total_seconds() > 0 else float('inf')
        print(f"Performance: New method is {speedup:.2f}x faster than original")

        # Save the results to file for detailed comparison if needed
        results_dir = Path("datasets/metrics/test")
        results_dir.mkdir(parents=True, exist_ok=True)

        original_results.to_csv(results_dir / "original_method_results.csv", index=False)
        new_results.to_csv(results_dir / "new_method_results.csv", index=False)
        print(f"Results saved to {results_dir}")

        return True

    except Exception as e:
        print(f"Error in test_worker_analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 0. Tests
    # test_worker_schedule()
    # communities = load_communities(configs[0])
    # test_worker_analysis()

    # # 1. Generate configurations and rules
    configs = generate_experiment_configurations()
    rules = generate_experiment_rules()

    # 2. Print a summary
    print("\nEXPERIMENT SETUP:", flush=True)
    print(f"  Number of Configurations: {len(configs)}", flush=True)
    print(f"  Number of Rules: {len(rules)}", flush=True)
    for rule in rules:
        print(f"  Rule: {rule}", flush=True)

    for cfg_dict in configs:
        cfg = cfg_dict["config"]
        print(f"\nConfiguration ID: {cfg_dict['id']}", flush=True)
        print(f"  Population: {cfg.num_agents}", flush=True)
        print(f"  Events: {cfg.num_events}", flush=True)
        print(f"  Agent Type: {cfg.agent_config.agent_type}", flush=True)
        if hasattr(cfg.agent_config, 'distribution'):
            print(f"  Distribution: {cfg.agent_config.distribution}", flush=True)

    # 3. Generate & Save communities ONCE per config
    for config_dict in configs:
        scheduler = Scheduler(config_dict['config'])
        communities = scheduler.create_communities()
        save_communities(config_dict, communities)
        print(f"Communities saved for {config_dict['id']}", flush=True)

    # 4. Parallel Event Scheduling
    communities = load_communities(configs[0])
    print("\nStarting parallel event scheduling...", flush=True)
    parallel_schedule(configs, rules)
    print("Event scheduling done.", flush=True)

    # 5. Parallel analytics
    print("\nStarting parallel analytics...")
    parallel_analysis(configs, rules)
    print("Analytics completed.")