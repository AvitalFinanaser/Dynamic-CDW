import datetime
import random
from typing import List, Tuple
import pandas as pd
from rules import *
from simulation.scheduler import Scheduler
from agents import *
from events import *
from simulation import solution_analysis as SA
from typing import List


class CDWSystem:
    """
    The Collaborative Document Writing (CDW) system consists of scheduler configured.
    This system processes event lists and applies rules to compute solutions,
    analyze metrics, and aggregate results.
    """

    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler
        self._seed = scheduler.config.random_seed
        random.seed(self._seed)
        self._instance = self._scheduler.schedule_single_instance()
        self._instances = self._scheduler.schedule_multiple_instances()

    # Single instance (A, E): tuple[list[Agent], EventList]
    def get_single_instances_data(self) -> tuple[list[Agent], EventList]:
        """
        Gets the single simulation data (community, event list) from the scheduler.
        """
        return self._instance

    def solution_1Condition(self, condition: Condition) -> list:
        """
        Solution to 1-Condition rule for single event list configuration
        :param condition: Condition rule with CSF and threshold
        :return: Solution as the list of paragraphs to be included in the document
        """
        Agents, Events = self._scheduler.schedule_single_instance()
        return condition.solution_1Condition(E=Events)

    def analyze_single_instance_satisfaction(self, rule: Condition) -> pd.DataFrame:
        """
        Analyzes satisfaction across for instance (A, E), calculating satisfaction metrics
        for each incremental set of events.
        Parameters:
        -----------
        rule: Condition
            The aggregation rule to generate solutions

        Returns:
        --------
        pd.DataFrame with satisfaction metrics:
        Sat_Sum, Sat_Avg, Sat_Sum_Normalized, Sat_Min, Sat_Max, Sat_Std
        """
        # Get instance (A,E) data
        community, event_list = self.get_single_instances_data()
        print(f"Event list: {event_list.events_df()}")
        print(f"Number of agents: {len(community)}")
        print(f"Number of events: {len(event_list.events)}")
        instance_results = SA.satisfaction_and_solutions(event_list=event_list, rule=rule)
        return instance_results

    # Multiple instances (A, E) * number of lists: List[tuple[List[Agent], EventList]]
    def get_multiple_instances_data(self) -> List[tuple[List[Agent], EventList]]:
        """
        Gets the basic simulation data (communities and their event lists)
        from the scheduler.
        """
        return self._instances

    def analyze_multiple_satisfaction(self, rule: Condition, instances_data=None) -> pd.DataFrame:
        """
        Analyzes satisfaction across all instances, calculating metrics
        for each incremental set of events and then aggregating across instances.
        Parameters:
        -----------
        rule: Condition
            The aggregation rule to generate solutions

        Returns:
        --------
        pd.DataFrame with aggregated (averaged) satisfaction metrics:
        Sat_Sum, Sat_Avg, Sat_Sum_Normalized, Sat_Min, Sat_Max, Sat_Std
        """
        # Get all instances
        if instances_data is None:
            instances_data = self.get_multiple_instances_data()

        # Process each instance
        all_results = []

        for community, event_list in instances_data:
            # Get satisfaction metrics for this instance
            instance_results = SA.satisfaction_and_solutions(event_list=event_list, rule=rule)
            all_results.append(instance_results)

        # Combine all results
        all_results_df = pd.concat(all_results, ignore_index=True)

        # Aggregate by event number across all instances
        aggregated_results = all_results_df.groupby('Event').agg({
            'Sat_Sum': 'mean',
            'Sat_Avg': 'mean',
            'Sat_Sum_Normalized': 'mean',
            'Sat_Min': 'mean',
            'Sat_Max': 'mean',
            'Sat_Std': 'mean'
        }).reset_index()

        return aggregated_results

    def analyze_multiple_stability(self, rule: Condition, instances_data=None) -> pd.DataFrame:
        """
        Analyzes stability metrics across all instances in the system using a specified rule.
        For each instance, calculates updates statistics for incremental solutions, then
        aggregates across instances to get average stability metrics per event number.

        This method uses the updates_and_solutions function from solution_analysis to process
        each individual event list, then combines and averages the results across all instances
        in the system.

        Parameters:
        -----------
        rule : Condition
            The aggregation rule to be applied for generating solutions

        Returns:
        --------
        pd.DataFrame: Contains aggregated stability metrics with columns:
            - Event: Event number (1 to n)
            - In_Update: Average number of in-updates across instances
            - Out_Update: Average number of out-updates across instances
            - Total_Average_Updates: Sum of average in and out updates
        """
        # Get all instances from our system's scheduler
        if instances_data is None:
            instances_data = self.get_multiple_instances_data()

        # Collect results from each instance
        all_results = []

        for community, event_list in instances_data:
            # Get stability metrics for this instance
            instance_stability = SA.updates_and_solutions(E=event_list, condition=rule)
            all_results.append(instance_stability)

        # Combine all instance results into a single DataFrame
        all_results_df = pd.concat(all_results, ignore_index=True)

        # Calculate average stability metrics across instances for each event number
        aggregated_stability = all_results_df.groupby('Event').agg({
            'Size': 'mean',
            'Stability': 'mean',
            'In_Update': 'mean',
            'Out_Update': 'mean',
            'Total_Updates': 'mean',
            'Total_In_Updates': 'mean',
            'Total_Out_Updates': 'mean'
        }).reset_index()

        # Calculate total updates as sum of in and out updates
        aggregated_stability['Total_Average_Updates'] = (
                aggregated_stability['In_Update'] +
                aggregated_stability['Out_Update']
        )

        return aggregated_stability

    def analyze_multiple_metrics(self, rule: Condition, instances_data=None) -> pd.DataFrame:
        """
        Combines satisfaction and stability metrics across all instances into a single analysis.
        This method leverages the existing satisfaction and stability analyses and merges
        their results based on the event number.

        Parameters:
        -----------
        rule : Condition
            The aggregation rule to be applied for generating solutions
        instances_data: Tuple[list[Agent], EventList]
            The datasets for instances (A, E) can come from outside source without scheduling

        Returns:
        --------
        pd.DataFrame with columns:
            - Event: Event number (1 to n)
            - Sat_Sum: Average total satisfaction across instances
            - Sat_Avg: Average mean satisfaction across instances
            - In_Update: Average number of in-updates across instances
            - Out_Update: Average number of out-updates across instances
            - Total_Average_Updates: Sum of average in and out updates
        """
        if instances_data is None:
            instances_data = self.get_multiple_instances_data()

        # Get both types of analyses
        satisfaction_results = self.analyze_multiple_satisfaction(rule, instances_data)
        stability_results = self.analyze_multiple_stability(rule, instances_data)
        # Merge the DataFrames on the 'Event' column
        combined_results = pd.merge(
            satisfaction_results,
            stability_results,
            on='Event',
            how='outer'
        )
        # Sort by Event number to ensure chronological order
        combined_results = combined_results.sort_values('Event').reset_index(drop=True)
        return combined_results

    def analyze_multiple_metrics_together(self, rule: Condition, instances_data=None) -> pd.DataFrame:
        """
        Combines satisfaction and stability metrics across all instances into a single analysis.
        This method leverages the existing satisfaction and stability analyses and merges
        their results based on the event number.

        Parameters:
        -----------
        rule : Condition
            The aggregation rule to be applied for generating solutions
        instances_data: Tuple[list[Agent], EventList]
            The datasets for instances (A, E) can come from outside source without scheduling

        Returns:
        --------
        pd.DataFrame with columns:
            - Event: Event number (1 to n)
            - Sat_Sum: Average total satisfaction across instances
            - Sat_Avg: Average mean satisfaction across instances
            - In_Update: Average number of in-updates across instances
            - Out_Update: Average number of out-updates across instances
            - Total_Average_Updates: Sum of average in and out updates
        """
        if instances_data is None:
            instances_data = self.get_multiple_instances_data()

        # Process each instance at a time
        sat_sta_results = []

        for community, event_list in instances_data:

            instance_results = []
            num_events = len(event_list.events)
            Ei = EventList()

            # Initiate all metrics indicators
            num_updates = 0
            num_in_updates = 0
            num_out_updates = 0
            current_solution = []


            # Process each incremental set of events
            for i in range(1, num_events + 1):
                Ei.add_event(event_list.events[i - 1])
                solution = rule.solution_1Condition(E=Ei)

                # Satisfaction

                # Weighted preference matrix - for each agent its Sat
                pref_matrix = SA.weighted_preference_matrix(event_list=Ei, final_solution=solution)
                stats = SA.calculate_satisfaction_statistics(pref_matrix)

                # Calculate active agents and normalized satisfaction
                Na_scores = [scores[4] for scores in pref_matrix.values()]
                num_agents = len([score for score in Na_scores if score != 0])
                sat_sum_normalized = stats['sum'] / num_agents if num_agents > 0 else 0

                # Stability
                update = 0
                in_update = 0
                out_update = 0
                next_solution = solution

                # Check if an update has occurred
                if current_solution != next_solution:
                    update = 1
                    num_updates += 1

                    # Determine if it is an in_update or out_update based on the length comparison
                    if len(next_solution) > len(current_solution):
                        in_update = 1  # Solution list length increased
                        num_in_updates += 1
                    elif len(next_solution) < len(current_solution):
                        out_update = 1  # Solution list length decreased
                        num_out_updates += 1

                # Update current solution to next solution
                current_solution = next_solution

                # Store metrics for this event
                instance_results.append({
                    'Event': i,
                    # Satisfaction metrics
                    'Sat_Sum': stats['sum'],
                    'Sat_Avg': stats['average'],
                    'Sat_Sum_Normalized': sat_sum_normalized,
                    'Sat_Min': stats['min'],
                    'Sat_Max': stats['max'],
                    'Sat_Std': stats['std_dev'],
                    # Stability metrics
                    'Size': len(current_solution),
                    'Update': update,
                    'In_Update': in_update,
                    'Out_Update': out_update,
                    'Total_Updates': num_updates,
                    'Total_In_Updates': num_in_updates,
                    'Total_Out_Updates': num_out_updates,
                    'Stability': (i - num_updates) / i
                })

            # Convert instance results to DataFrame and add to collection
            sat_sta_results.append(pd.DataFrame(instance_results))

        # Combine all results
        all_results_df = pd.concat(sat_sta_results, ignore_index=True)

        # Aggregate by event number across all instances

        aggregated_results = all_results_df.groupby('Event').agg({
            'Sat_Sum': 'mean',
            'Sat_Avg': 'mean',
            'Sat_Sum_Normalized': 'mean',
            'Sat_Min': 'mean',
            'Sat_Max': 'mean',
            'Sat_Std': 'mean',
            'Size': 'mean',
            'Stability': 'mean',
            'In_Update': 'mean',
            'Out_Update': 'mean',
            'Total_Updates': 'mean',
            'Total_In_Updates': 'mean',
            'Total_Out_Updates': 'mean'
        }).reset_index()

        aggregated_results['Total_Average_Updates'] = (
                aggregated_results['In_Update'] +
                aggregated_results['Out_Update']
        )

        return aggregated_results

    def plot_stability_analysis(self, rule: Condition, analysis_results: pd.DataFrame):
        """
        Creates a publication-quality stacked area plot showing solution updates over time.
        Takes a DataFrame with In_Update and Out_Update columns and creates a cumulative
        visualization.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure
        plt.figure(figsize=(8, 5))

        # Ensure we're working with numeric data
        events = analysis_results['Event'].astype(float).values
        in_updates = analysis_results['In_Update'].astype(float).cumsum().values
        out_updates = analysis_results['Out_Update'].astype(float).cumsum().values

        # Create the stacked plot - note we're passing the data differently
        plt.stackplot(events,
                      [in_updates, out_updates],
                      labels=['In Updates', 'Out Updates'],
                      colors=['#0077BB', '#EE7733'],
                      alpha=0.7)

        # Basic styling for academic publication
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Events', fontsize=11)
        plt.ylabel('Cumulative Updates', fontsize=11)
        plt.title(f'{rule.__str__()}', fontsize=12)
        plt.legend(fontsize=10)

        # Save the plot
        n = f"Number of agents: {self._scheduler.config.num_agents}"
        t = f"{self._scheduler.config.num_events}"
        n_lists = f"{self._scheduler.config.num_lists}"
        agent_t = f"Agent type: {self._scheduler.config.agent_config.agent_type.value}"
        save_path = f'C:/Users/avita/Desktop/_stability.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stability analysis plot saved at: {save_path}")

    def plot_satisfaction_analysis(self, rule: Condition, analysis_results: pd.DataFrame):
        """
        Creates a publication-quality line plot showing satisfaction metrics over time.
        Takes a DataFrame with Sat_Sum and Sat_Avg columns and creates a visualization
        line chart.
        """
        import matplotlib.pyplot as plt

        colors = {
            'total': '#1f77b4',
            'avg': '#7f7f7f'
        }

        # Create figure with academic dimensions
        plt.figure(figsize=(10, 6))

        # Set minimal style with dotted grid
        plt.grid(True, which='both', axis='both',
                 linestyle=':', linewidth=0.5,
                 color='gray', alpha=0.5)

        # Sort data by Event number for proper line ordering
        analysis_results = analysis_results.sort_values('Event')

        # Plot each metric with clean styling
        plt.plot(analysis_results['Event'],
                 analysis_results['Sat_Sum'],
                 color=colors['total'],  # Professional blue
                 linewidth=1.5,
                 label='Sum(Sat)')

        plt.plot(analysis_results['Event'],
                 analysis_results['Sat_Avg'],
                 color=colors['avg'],  # Professional gray
                 linewidth=1.5,
                 label='Avg(Sat)')

        # Customize axes
        plt.ylim(0, 1.1)
        plt.yticks([i / 5 for i in range(6)])  # 0.0 to 1.0 by 0.2
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Add labels
        plt.xlabel('Number of Events', fontsize=11)
        plt.ylabel('Satisfaction Score', fontsize=11)
        plt.title(f'{rule.__str__()}', fontsize=12, pad=10)

        # Enhanced legend
        plt.legend(loc='upper right', fontsize=10,
                   framealpha=0.9, edgecolor='lightgray')

        # Save plot
        config = self._scheduler.config
        filename = f"satisfaction_n{config.num_agents}_e{config.num_events}.png"
        save_path = f'C:/Users/avita/Desktop/_satisfaction.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Satisfaction analysis plot saved at: {save_path}")

    def plot_satisfaction_normalized_analysis(self, rule: Condition, analysis_results: pd.DataFrame):
        """
        Creates a publication-quality line plot showing satisfaction metrics over time,
        with normalized satisfaction measures for better comparison.
        """
        import matplotlib.pyplot as plt

        # Create figure
        plt.figure(figsize=(10, 7))
        ax = plt.gca()

        # Sort data and normalize sum satisfaction
        analysis_results = analysis_results.sort_values('Event')
        num_agents = self._scheduler.config.num_agents
        normalized_sum = analysis_results['Sat_Sum'] / num_agents  # Normalize by number of agents

        # Plot normalized metrics
        line1 = ax.plot(analysis_results['Event'],
                        normalized_sum,
                        color='#1f77b4',
                        linewidth=1.5,
                        label='Sum(Sat)')

        line2 = ax.plot(analysis_results['Event'],
                        analysis_results['Sat_Avg'],
                        color='#7f7f7f',
                        linewidth=1.5,
                        label='Avg(Sat)')

        # Style the plot
        ax.grid(True, which='both', axis='both',
                linestyle=':', linewidth=0.5,
                color='gray', alpha=0.5)

        # Set axis limits and ticks for normalized scale
        ax.set_ylim(0, 1.1)
        ax.set_yticks([i / 5 for i in range(6)])
        ax.tick_params(axis='both', labelsize=10)

        # Add labels
        ax.set_xlabel('Number of Events', fontsize=11)
        ax.set_ylabel('Satisfaction Score', fontsize=11)
        ax.set_title(f'{rule.__str__()}', fontsize=12, pad=10)

        # Place legend below plot
        legend = ax.legend(handles=[line1[0], line2[0]],
                           labels=['Sum(Sat)', 'Avg(Sat)'],
                           loc='upper center',
                           bbox_to_anchor=(0.5, -0.15),
                           ncol=2,
                           frameon=False,
                           fontsize=10)

        plt.tight_layout()
        save_path = f'C:/Users/avita/Desktop/_satisfaction_N.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # Getters and Setters
    def get_scheduler(self):
        return self._scheduler

    def set_scheduler(self, scheduler):
        self._scheduler = scheduler

    def get_seed(self):
        return self._seed

    def set_seed(self, seed):
        self._seed = seed
        random.seed(self._seed)