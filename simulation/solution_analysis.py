# First, at the very start of your script, add these lines:
import datetime

import matplotlib
from tabulate import tabulate
import random
import matplotlib.pyplot as plt
import pandas as pd
from rules import *
from agents import *
from events import *
from paragraphs import Paragraph, ParagraphWithScore
import numpy as np

matplotlib.use('TkAgg')  # This sets the backend to TkAgg which usually works well
random.seed(42)


## A solution functions

# R(T) = list of paragraphs
def solution_1Condition(E: EventList, condition: Condition) -> list:
    """
    Solution to 1-Condition rule
    :param E: Event list Object
    :param condition: Condition rule with CSF and threshold
    :return: Solution as the list of paragraphs to be included in the document
    """
    solution = []

    if isinstance(condition, StaticCondition):

        for p in E.P():  # For every paragraph in P(E)
            include = condition.compliance(E=E, p=p)
            # Include if condition holds, else remove
            if p in solution and include == False:
                solution.remove(p)
            elif p not in solution and include == True:
                solution.append(p)

    elif isinstance(condition, HarshDynamicCondition):

        for p in E.P():  # For every paragraph in P(E)
            include = condition.compliance(E=E, p=p)
            # Include if condition holds, else remove
            if p in solution and include == False:
                solution.remove(p)
            elif p not in solution and include == True:
                solution.append(p)

    elif isinstance(condition, SmoothDynamicCondition):

        if condition.dynamic_property == "events":

            for t, event in enumerate(E.events, start=1):
                # Sublist of the first t events
                E_t = EventList()
                E_t.events = E.E_i(t)
                for p in E_t.P():
                    include = condition.compliance(E=E_t, p=p, t=t)
                    # Include if condition holds, else remove
                    if p in solution and include == False:
                        solution.remove(p)
                    elif p not in solution and include == True:
                        solution.append(p)

    else:
        raise ValueError("Unknown condition type")

    return solution


# S(E) = list of paragraphs names
def S_E(sol: list[Paragraph]):
    """
    Solution list of paragraphs included in document
    :param sol: List of Paragraph objects
    :return: List of the paragraphs names (headlines)
    """
    S_E = []
    for p in sol:
        S_E.append(p.name)
    return S_E


# S(E)_text = list of paragraphs text - content of document
def S_E_texts(sol: list[Paragraph]):
    """
    Solution list of paragraphs included in document
    :param sol: List of Paragraph objects
    :return: List of the paragraphs names (headlines)
    """
    S_E = []
    for p in sol:
        S_E.append(p.text)
    return S_E


# Sub solutions S*
def solutions(E: EventList, condition: Condition):
    """
    S* = List of temporary solutions after each event.
    :param E: Event list Object
    :param condition: Condition rule with CSF and threshold
    :return: List of sub-solutions after each event occurrence
    """
    sol_list = []  # List of temporary solutions
    for i in range(1, len(E.events) + 1):  # For any sub-list
        event_list_i = EventList()
        event_list_i.events = E.E_i(i)  # sub-list of E from start until include event i
        s_i = condition.solution_1Condition(E=event_list_i)
        sol_list.append(s_i)
    return sol_list


def print_solutions_names(sol_list: list):
    """
    Displays solutions after each iteration in names format
    :param sol_list: List of sub-solutions after each event occurrence
    """
    for i, sol in enumerate(sol_list, start=1):
        S_E = [p.name for p in sol]
        print(f"Solution after {i} events: {S_E}")


def print_solutions_text(sol_list: list[Paragraph]):
    """
    Displays solutions paragraph text after each iteration in text format
    :param sol_list: List of sub-solutions after each event occurrence
    """
    for i, sol in enumerate(sol_list, start=1):
        S_E = [p.text for p in sol]
        print(f"Solution after {i} events: {S_E}")


# ~ Stability metrics

# Summary
def updates_and_solutions(E: EventList, condition: Condition):
    """
    Track updates made in the solution after each event.
    This function creates a detailed timeline showing:
    1. What changes occurred after each event (additions or removals)
    2. The size of the solution at each point
    3. A running count of total updates
    :param E: list of events
    :param condition: Condition rule with CSF and threshold
    :return: DataFrame with each event, the solution after its arrival, indicators for updates (in_update, out_update), and total updates.
    """
    results = pd.DataFrame(columns=['Event', 'Solution', 'Size', 'Update', 'In_Update', 'Out_Update',
                                    'Total Updates', 'Total In Updates', 'Total Out Updates', 'Stability'])
    sol_list = solutions(E=E, condition=condition)
    num_updates = 0
    num_in_updates = 0
    num_out_updates = 0
    current_solution = []

    for i, sol in enumerate(sol_list, start=1):
        update = 0
        in_update = 0
        out_update = 0
        next_solution = sol

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

        # Append the results to the DataFrame
        row = {
            'Event': i,
            'Solution': ', '.join([p.name for p in current_solution]),
            'Size': len(current_solution),
            'Update': update,
            'In_Update': in_update,
            'Out_Update': out_update,
            'Total_Updates': num_updates,
            'Total_In_Updates': num_in_updates,
            'Total_Out_Updates': num_out_updates,
            'Stability': (i - num_updates) / i
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    return results


# aggregation
def aggregate_types_updates(event_lists: list[EventList], condition: Condition):
    """
    Average number of in_updates, out_updates, and total updates given list of event lists according to a condition.

    :param event_lists: List of event list objects
    :param condition: Condition rule with CSF and threshold
    :return: Data frame with event number, average in_updates, out_updates, and total updates.
    """
    # Collect all results in a list
    all_results = []

    for event_list in event_lists:
        print("Next event list")
        df = updates_and_solutions(E=event_list, condition=condition)
        all_results.append(df)

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Group by 'Event' and calculate the average 'In_Update', 'Out_Update'
    avg_updates = all_results_df.groupby('Event')[['In_Update', 'Out_Update']].mean().reset_index()

    # Calculate total average updates as the sum of in_updates and out_updates
    avg_updates['Total_Average_Updates'] = avg_updates['In_Update'] + avg_updates['Out_Update']

    return avg_updates


# Plot 1 - stability of solutions over time. Using avg_updates_df = aggregate_types_updates
def plot_stacked_area_cumulative_updates(avg_updates_df, condition: Condition):
    """
    Plot a stacked area chart of cumulative in_updates and out_updates.
    :param condition: Condition rule with CSF and threshold.
    :param avg_updates_df: DataFrame containing average updates per event.
    """
    # Ensure events and updates are numeric
    avg_updates_df['Event'] = pd.to_numeric(avg_updates_df['Event'], errors='coerce')
    avg_updates_df['In_Update'] = pd.to_numeric(avg_updates_df['In_Update'], errors='coerce')
    avg_updates_df['Out_Update'] = pd.to_numeric(avg_updates_df['Out_Update'], errors='coerce')

    # Compute cumulative sums for in and out updates
    avg_updates_df['Cumulative_In_Update'] = avg_updates_df['In_Update'].cumsum()
    avg_updates_df['Cumulative_Out_Update'] = avg_updates_df['Out_Update'].cumsum()

    # Extract data from the DataFrame
    events = avg_updates_df['Event'].values
    in_updates = avg_updates_df['Cumulative_In_Update'].values
    out_updates = avg_updates_df['Cumulative_Out_Update'].values

    # Create stacked area plot
    plt.figure(figsize=(10, 6))

    # Set more formal color palette
    in_color = '#1f77b4'  # Light blue
    out_color = '#7f7f7f'  # Light gray

    # Add gridlines
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    # Create the stacked area chart using stackplot
    plt.stackplot(events, in_updates, out_updates, labels=['In Updates', 'Out Updates'], colors=[in_color, out_color])

    # Add labels, title, and legend

    # plt.xlabel('Number of Events', fontsize=12)
    # plt.ylabel('Cumulative Updates', fontsize=12)
    # plt.title(f'Cumulative In and Out Updates vs. Number of Events\n({condition})', fontsize=14, pad=20)
    # plt.legend(loc='upper left', fontsize=10)

    # Set tick sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Display the plot with tighter layout
    plt.tight_layout()
    plt.show()


# ~ Satisfaction metrics


def preference_matrix(event_list, final_solution):
    """
    Calculate the preference matrix for each agent
    :param event_list: List of events.
    :param final_solution: List of paragraph representing the final solution.
    :return: A dictionary with agent IDs as keys and a list [TP, TN, FP, FN] as values.
    """
    agents, paragraphs, summary_matrix = event_list.sum()
    final_solution_ids = [p.paragraph_id for p in final_solution]
    # Initialize the result matrix with zeros: [TP, TN, FP, FN] for each agent
    preference_matrix = {agent: [0, 0, 0, 0] for agent in agents}

    for j, paragraph in enumerate(paragraphs):
        in_final_solution = paragraph in final_solution_ids

        for i, agent in enumerate(agents):
            vote = summary_matrix[i][j]

            if vote == 0:
                continue  # Skip if the vote is abstention or non-voting

            if vote == 1:
                if in_final_solution:
                    preference_matrix[agent][0] += 1  # TP
                else:
                    preference_matrix[agent][2] += 1  # FP

            elif vote == -1:
                if in_final_solution:
                    preference_matrix[agent][3] += 1  # FN
                else:
                    preference_matrix[agent][1] += 1  # TN
    return preference_matrix


def print_preference_matrix(preference_matrix, event_list: EventList):
    """
    Print the preference matrix as a table
    :param event_list: list of events
    :param preference_matrix: Matrix that represents the agent's preferences
    """
    # Create table headers
    table_headers = ['Agent', 'TP', 'TN', 'FP', 'FN']

    # Create table data
    table_data = [[Agent.get_agent_name_by_id(agent_id, event_list.A_E())] + scores for agent_id, scores in
                  preference_matrix.items()]

    # Print table
    print(tabulate(table_data, headers=table_headers, tablefmt="grid", stralign="center"))


def weighted_preference_matrix(event_list, final_solution):
    """
    Calculate the weighted preference matrix for each agent including satisfaction scores

    Parameters:
    event_list: List of events
    final_solution: List of paragraphs representing the final solution

    Returns:
    Dictionary with agent IDs as keys and lists [TP, TN, FP, FN, N_a, Sat_a, Sat_a_avg] as values
    where Sat_a is the satisfaction score for agent a
    """
    agents, paragraphs, summary_matrix = event_list.sum()
    final_solution_ids = [p.paragraph_id for p in final_solution]

    # Initialize the result matrix with zeros: [TP, TN, FP, FN, N_a, Sat_a]
    weighted_matrix = {agent: [0, 0, 0, 0, 0, 0.0] for agent in agents}

    # First pass: Calculate TP, TN, FP, FN to determine N_a
    for j, paragraph in enumerate(paragraphs):
        in_final_solution = paragraph in final_solution_ids

        for i, agent in enumerate(agents):
            vote = summary_matrix[i][j]

            if vote == 0:
                continue  # Skip if the vote is abstention or non-voting

            if vote == 1:
                if in_final_solution:
                    weighted_matrix[agent][0] += 1  # TP
                else:
                    weighted_matrix[agent][2] += 1  # FP

            elif vote == -1:
                if in_final_solution:
                    weighted_matrix[agent][3] += 1  # FN
                else:
                    weighted_matrix[agent][1] += 1  # TN

    # Second pass: Calculate N_a and Sat_a for each agent
    for agent in agents:
        # Calculate N_a as sum of all preferences
        N_a = sum(weighted_matrix[agent][:4])  # Sum of TP, TN, FP, FN
        weighted_matrix[agent][4] = N_a

        # Calculate Sat_a as sum of correct votes (TP + TN) divided by N_a
        if N_a > 0:  # Avoid division by zero
            correct_votes = weighted_matrix[agent][0] + weighted_matrix[agent][1]  # TP + TN
            Sat_a = (correct_votes * (1.0 / N_a))  # Each correct vote contributes 1/N_a
            weighted_matrix[agent][5] = Sat_a

    return weighted_matrix


def print_weighted_preference_matrix(weighted_matrix, event_list: EventList):
    """
    Print the weighted preference matrix as a formatted table with all metrics

    Parameters:
    weighted_matrix: Dictionary with agent IDs as keys and lists [TP, TN, FP, FN, N_a, Sat_a] as values
    event_list: Event list. Require for community agents.
    """
    # Import tabulate if not already imported
    from tabulate import tabulate

    # Create table headers with the new metrics
    table_headers = ['Agent', 'TP', 'TN', 'FP', 'FN', 'N_a', 'Satisfaction']

    # Create table data with formatted values
    table_data = []
    for agent_id, scores in weighted_matrix.items():
        # Get agent name
        agent_name = Agent.get_agent_name_by_id(agent_id, event_list.A_E())

        # Format the satisfaction score to 3 decimal places
        # First 5 values are integers (TP, TN, FP, FN, N_a), last value is float (Sat_a)
        formatted_scores = [
            *scores[:5],  # Keep integer format for first 5 values
            f"{scores[5]:.3f}"  # Format satisfaction score to 3 decimal places
        ]

        # Combine agent name with scores
        row = [agent_name] + formatted_scores
        table_data.append(row)

    # Sort table data by agent name for consistent display
    table_data.sort(key=lambda x: x[0])

    # Print table with additional formatting
    print("\nWeighted Preference Matrix:")
    print(tabulate(
        table_data,
        headers=table_headers,
        tablefmt="grid",
        stralign="center",
        numalign="right"  # Right-align numbers for better readability
    ))


def calculate_satisfaction_statistics(weighted_matrix):
    """
    Calculate summary statistics for  agents satisfaction scores from the weighted preference matrix.

    Parameters:
    weighted_matrix: Dictionary with agent IDs as keys and lists [TP, TN, FP, FN, N_a, Sat_a] as values.

    Returns:
    Dictionary containing various satisfaction statistics:
        - individual_scores: List of all individual satisfaction scores
        - sum: Total sum of satisfaction scores
        - average: Mean satisfaction score
        - median: Median satisfaction score
        - std_dev: Standard deviation of satisfaction scores
        - min: Minimum satisfaction score
        - max: Maximum satisfaction score
        - quartiles: Dictionary with Q1, Q2 (median), and Q3
    """
    # Extract satisfaction scores- Sat_a (index 5 in each agent's list)
    satisfaction_scores = [scores[5] for scores in weighted_matrix.values()]
    scores_array = np.array(satisfaction_scores)

    # Calculate quartiles
    q1, q2, q3 = np.percentile(scores_array, [25, 50, 75])

    # Compile statistics into a structured dictionary
    statistics = {
        'individual_scores': satisfaction_scores,  # Keep raw scores for detailed analysis
        'sum': np.sum(scores_array),  # Total satisfaction
        'average': np.mean(scores_array),  # Mean satisfaction
        'median': q2,  # Median (Q2)
        'std_dev': np.std(scores_array),  # Standard deviation
        'min': np.min(scores_array),  # Minimum score
        'max': np.max(scores_array),  # Maximum score
        'quartiles': {
            'Q1': q1,  # First quartile
            'Q2': q2,  # Second quartile (median)
            'Q3': q3  # Third quartile
        }
    }

    return statistics


def satisfaction_and_solutions(event_list: EventList, rule: Condition) -> pd.DataFrame:
    """
    Calculates satisfaction metrics for each incremental solution in an event list.
    For each event i, takes events 1 to i and calculates satisfaction metrics
    using the provided rule.

    Parameters:
    -----------
    event_list: EventList
        The event list to analyze
    rule: Condition
        The aggregation rule to generate solutions

    Returns:
    --------
    pd.DataFrame with columns:
        - Event: Event number (1 to t)
        - Sat_Sum: Sum of total satisfaction for events 1 to i
        - Sat_Avg: Average of agents satisfaction for events 1 to i
        - Sat_Sum_Normalized: Proportion of total possible satisfaction (sum/num_agents)
        - Sat_Min: Minimum satisfaction score among active agents
        - Sat_Max: Maximum satisfaction score among active agents
        - Sat_Std: Standard deviation of satisfaction scores
    """
    results = []
    num_events = len(event_list.events)
    Ei = EventList()
    # Process each incremental set of events
    for i in range(1, num_events + 1):
        # Sub event list with i events - E_i
        # Ei = EventList()
        # Ei.events = event_list.E_i(i=i)
        Ei.add_event(event_list.events[i - 1])
        # print(Ei.events_df()) # -
        # Solution of E_i using given rule
        solution = rule.solution_1Condition(E=Ei)
        # Weighted preference matrix - for each agent its Sat
        start = datetime.datetime.now()
        pref_matrix = weighted_preference_matrix(event_list=Ei, final_solution=solution)
        # print_weighted_preference_matrix(pref_matrix, Ei)
        start = datetime.datetime.now()
        stats = calculate_satisfaction_statistics(pref_matrix)
        Na_scores = [scores[4] for scores in pref_matrix.values()]

        # The number of active agents = agents with number of preference vote > 0
        num_agents = len([score for score in Na_scores if score != 0])
        # sat_sum_normalized - the sum of agents sat score divided with active agents (proportion of satisfaction from max possible)
        sat_sum_normalized = stats['sum'] / num_agents if num_agents > 0 else 0
        # print(f"Sum_a: {stats['sum']}, Sat_normalized: {sat_sum_normalized}, Sat_avg: {stats['average']}")

        # Store satisfaction results for E_i for all agents
        results.append({
            'Event': i,
            'Sat_Sum': stats['sum'],
            'Sat_Avg': stats['average'],
            'Sat_Sum_Normalized': sat_sum_normalized,  # Normalize by number of agents
            'Sat_Min': stats['min'],  # Add minimum satisfaction
            'Sat_Max': stats['max'],  # Add maximum satisfaction
            'Sat_Std': stats['std_dev'],  # Add standard deviation
        })

    return pd.DataFrame(results)


def filter_active_agents_in_matrix(preference_matrix, active_agents):
    """
    Filters the preference matrix, keeping only rows corresponding to active agents.
    :param preference_matrix: Dictionary of agent IDs as keys and their [TP, TN, FP, FN] preferences as values.
    :param active_agents: List of active agent objects.
    :return: A filtered preference matrix containing only active agents.
    """
    # Create a mapping of active agent IDs
    active_agent_ids = {agent.agent_id for agent in active_agents}

    # Filter the preference matrix to keep only active agents
    filtered_matrix = {agent_id: preference_matrix[agent_id] for agent_id in preference_matrix if
                       agent_id in active_agent_ids}

    return filtered_matrix


def event_satisfaction(event_list: EventList, condition: Condition):
    """
    This function creates a dataframe with each event and the satisfaction of agents using the given condition rule.
    :param event_list: Event list
    :param condition: Condition rule with CSF and threshold
    :return: A DataFrame with the number of events and community satisfaction scores.
    """
    results = []
    sol_list = solutions(E=event_list, condition=condition)

    for i, sol in enumerate(sol_list, start=1):
        # Step 1: Create a list of relevant agents for E_i
        Ei = EventList()
        Ei.events = event_list.E_i(i=i)
        relevant_agents = Ei.activeAgents()

        # Step 2: Create preference matrix for sub-list of events
        preferenceMatrix = preference_matrix(event_list=Ei, final_solution=sol)

        # Step 3: Filter agents in preference matrix
        filtered_matrix = filter_active_agents_in_matrix(preference_matrix=preferenceMatrix,
                                                         active_agents=relevant_agents)

        # Step 4: Calculate the utility of each agent
        scores_matrix = agent_satisfaction_utility(filtered_matrix)

        # Step 5: Aggregate preferences
        accuracies = [score[4] for score in scores_matrix.values()]  # Extract "Accuracy" (index 4)
        b_accuracies = [score[5] for score in scores_matrix.values()]  # Extract "Balanced Accuracy" (index 5)

        # Calculate averages and minimums directly
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        avg_b_accuracy = sum(b_accuracies) / len(b_accuracies) if b_accuracies else 0
        min_accuracy = min(accuracies) if accuracies else 0
        min_b_accuracy = min(b_accuracies) if b_accuracies else 0

        # Append the results to the DataFrame
        results.append({
            'Event': i,
            'Accuracy(avg)': avg_accuracy,
            'B-Accuracy(avg)': avg_b_accuracy,
            'Accuracy(min)': min_accuracy,
            'B-Accuracy(min)': min_b_accuracy
        })

    results_df = pd.DataFrame(results)
    return results_df


# Aggregations


def aggregate_event_satisfaction(event_lists: list[EventList], condition: Condition):
    """
     Aggregated satisfaction based on the utility matrix. This function creates a dataframe with each event and the average satisfaction of agents
     across the event lists using the given condition rule.
    :param event_lists: List of event list objects
    :param condition: Condition rule with CSF and threshold
    :return: A DataFrame with the number of events and aggregated satisfaction scores.
    """
    # Collect all results in a list
    all_results = []

    for event_list in event_lists:
        print("Next event list")
        df = event_satisfaction(event_list=event_list, condition=condition)
        all_results.append(df)

    # Concatenate all DataFrames at once
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Group by 'Event' to calculate the average and min for each event across all event lists
    aggregated_results = all_results_df.groupby('Event').agg({
        'Accuracy(avg)': 'mean',
        'B-Accuracy(avg)': 'mean',
        'Accuracy(min)': 'mean',
        'B-Accuracy(min)': 'mean'
    }).reset_index()

    return aggregated_results


def aggregate_updates_in_out(event_lists: list, condition: Condition):
    """
    Average number of updates given a list of event lists according to rule.
    :param event_lists: List of event list objects.
    :param condition: Condition rule with CSF and threshold
    :return: DataFrame with event number and average updates for 'In_Update' and 'Out_Update'.
    """
    # Collect all results in a list
    all_results = []
    for event_list in event_lists:
        print("Next event list")
        df = updates_and_solutions(E=event_list, condition=condition)
        all_results.append(df)

    # Concatenate all DataFrames at once
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Group by 'Event' and calculate the average for 'In_Update' and 'Out_Update'
    avg_updates = all_results_df.groupby('Event')[['In_Update', 'Out_Update']].mean().reset_index()

    return avg_updates


# Plots

# Plot 1 - stability of solutions over time
# avg_updates_df = aggregate_types_updates
def plot_stacked_area_cumulative_updates(avg_updates_df, condition: Condition):
    """
    Plot a stacked area chart of cumulative in_updates and out_updates.
    :param condition: Condition rule with CSF and threshold.
    :param avg_updates_df: DataFrame containing average updates per event.
    """
    # Ensure events and updates are numeric
    avg_updates_df['Event'] = pd.to_numeric(avg_updates_df['Event'], errors='coerce')
    avg_updates_df['In_Update'] = pd.to_numeric(avg_updates_df['In_Update'], errors='coerce')
    avg_updates_df['Out_Update'] = pd.to_numeric(avg_updates_df['Out_Update'], errors='coerce')

    # Compute cumulative sums for in and out updates
    avg_updates_df['Cumulative_In_Update'] = avg_updates_df['In_Update'].cumsum()
    avg_updates_df['Cumulative_Out_Update'] = avg_updates_df['Out_Update'].cumsum()

    # Extract data from the DataFrame
    events = avg_updates_df['Event'].values
    in_updates = avg_updates_df['Cumulative_In_Update'].values
    out_updates = avg_updates_df['Cumulative_Out_Update'].values

    # Create stacked area plot
    plt.figure(figsize=(10, 6))

    # Set more formal color palette
    in_color = '#1f77b4'  # Light blue
    out_color = '#7f7f7f'  # Light gray

    # Add gridlines
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    # Create the stacked area chart using stackplot
    plt.stackplot(events, in_updates, out_updates, labels=['In Updates', 'Out Updates'], colors=[in_color, out_color])

    # Add labels, title, and legend

    # plt.xlabel('Number of Events', fontsize=12)
    # plt.ylabel('Cumulative Updates', fontsize=12)
    # plt.title(f'Cumulative In and Out Updates vs. Number of Events\n({condition})', fontsize=14, pad=20)
    # plt.legend(loc='upper left', fontsize=10)

    # Set tick sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Display the plot with tighter layout
    plt.tight_layout()
    plt.show()


def plot_average_updates_double_bar_chart(event_lists: list[EventList], condition: Condition):
    """
    Create a double bar chart to visualize the average number of in and out updates across events.
    :param event_lists: List of event lists to be averaged.
    :param condition: Condition rule with CSF and threshold
    """
    # Get the average updates DataFrame
    avg_updates = aggregate_updates_in_out(event_lists, condition)

    # Plotting
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = avg_updates['Event']

    # Bar plots for 'In_Update' and 'Out_Update'
    plt.bar(index - bar_width / 2, avg_updates['In_Update'], bar_width, label='In Update', color='b')
    plt.bar(index + bar_width / 2, avg_updates['Out_Update'], bar_width, label='Out Update', color='r')

    # Labels and title
    plt.xlabel('Event Number')
    plt.ylabel('Average Number of Updates')
    plt.title('Average In and Out Updates Across Events')
    plt.xticks(index)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_aggregated_event_satisfaction(event_lists: list[EventList], condition: Condition):
    """
    Create a plot with line charts for each aggregation method (avg, min) over events.
    The x-axis is the number of events, and the y-axis is the B-Accuracy score.
    :param event_lists: List of event lists.
    :param condition: Condition rule with CSF and threshold.
    """
    aggregated_df = aggregate_event_satisfaction(event_lists=event_lists, condition=condition)
    agg_avg = aggregated_df['B-Accuracy(avg)']
    agg_min = aggregated_df['B-Accuracy(min)']
    events = aggregated_df['Event']

    # Set more formal color palette for the plot
    avg_color = '#1f77b4'  # Blue for avg
    min_color = '#7f7f7f'  # Gray for min

    # Add gridlines for better visibility
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    # Plot the avg and min B-Accuracy over events
    plt.plot(events, agg_avg, label='B-Accuracy(avg)', color=avg_color, linewidth=2)
    plt.plot(events, agg_min, label='B-Accuracy(min)', color=min_color, linewidth=2)

    # Add labels and title
    # plt.xlabel('Number of Events', fontsize=12)
    # plt.ylabel('B-Accuracy Score', fontsize=12)
    # plt.title(f'B-Accuracy Score over Events\n({condition})', fontsize=14, pad=20)
    # Add legend to differentiate between avg and min lines
    # plt.legend(loc='upper left', fontsize=10)

    # Set tick sizes for readability
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Display the plot with a tight layout
    plt.tight_layout()
    plt.show()


def plot_avg_updates(avg_updates, condition: Condition):
    """
    Plot simulation
    :param avg_updates: DataFrame containing average updates per event.
    :param condition: Condition rule with CSF and threshold.
    """
    plt.plot(avg_updates['Event'], avg_updates['Total Updates'],
             label=f'{condition.CSF}, threshold={condition.threshold}')
    plt.xlabel('Number of Events')
    plt.ylabel('Average Total Updates')
    plt.title('Average Total Updates vs. Number of Events')
    plt.legend()
    plt.show()


def plot_stacked_updates(avg_updates_df, condition: Condition, event_gap=20):
    """
     Plot a stacked bar chart of in_updates and out_updates vs. number of events.
    :param event_gap: Interval for displaying bars (default is every 20 events).
    :param condition: Condition rule with CSF and threshold.
    :param avg_updates_df: DataFrame containing average updates per event.
    """
    # Filter data to show only events with the specified gap
    filtered_df = avg_updates_df[avg_updates_df['Event'] % event_gap == 0]

    # Extract data from the filtered DataFrame
    events = filtered_df['Event']
    in_updates = filtered_df['In_Update']
    out_updates = filtered_df['Out_Update']
    total_updates = filtered_df['Total_Average_Updates']

    # Set the position of the bars on the x-axis
    bar_width = 0.35
    index = np.arange(len(events))

    # Create stacked bar chart for insertions and removals
    plt.figure(figsize=(8, 6))
    plt.bar(index, in_updates, bar_width, label='In update', color='lightblue')
    plt.bar(index, out_updates, bar_width, bottom=in_updates, label='Out update', color='lightcoral')

    # Add total average updates line
    plt.plot(index, total_updates, label='Total Average Updates', color='black', linewidth=2, marker='o')

    # Add labels, title, and legend
    plt.xlabel('Number of Events')
    plt.ylabel('Average Updates')
    plt.title(f'Average In and Out Updates vs. Number of Events ({condition.__str__()})')

    # Set ticks on x-axis for filtered events
    plt.xticks(index, events.astype(int), rotation=45)  # Show bars only for every event with the specified gap
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


# Updated function to calculate cumulative totals for stacked bar chart
def plot_stacked_bar_cumulative_updates(avg_updates_df, condition: Condition, event_gap=20):
    """
     Plot a stacked bar chart of cumulative in_updates and out_updates with bars only for every nth event.
    :param event_gap: Interval for displaying bars (default is every 20 events).
    :param condition: Condition rule with CSF and threshold.
    :param avg_updates_df: DataFrame containing average updates per event.
    """
    # Compute cumulative sums for in and out updates
    avg_updates_df['Cumulative_In_Update'] = avg_updates_df['In_Update'].cumsum()
    avg_updates_df['Cumulative_Out_Update'] = avg_updates_df['Out_Update'].cumsum()
    avg_updates_df['Cumulative_Total_Updates'] = avg_updates_df['Cumulative_In_Update'] + avg_updates_df[
        'Cumulative_Out_Update']

    # Filter data to show only events with the specified gap
    filtered_df = avg_updates_df[avg_updates_df['Event'] % event_gap == 0]

    # Extract data from the filtered DataFrame
    events = filtered_df['Event']
    in_updates = filtered_df['Cumulative_In_Update']
    out_updates = filtered_df['Cumulative_Out_Update']
    total_updates = filtered_df['Cumulative_Total_Updates']

    # Set the position of the bars on the x-axis
    bar_width = 0.35
    index = np.arange(len(events))

    # Create stacked bar chart for cumulative insertions and removals
    plt.figure(figsize=(8, 6))
    plt.bar(index, in_updates, bar_width, label='In updates', color='lightblue')
    plt.bar(index, out_updates, bar_width, bottom=in_updates, label='Out updates', color='lightcoral')

    # Add cumulative total updates line
    plt.plot(index, total_updates, label='Cumulative Total Updates', color='black', linewidth=2, marker='o')

    # Add labels, title, and legend
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Updates')
    plt.title(f'Cumulative In and Out Updates vs. Number of Events ({condition.__str__()})')

    # Set ticks on x-axis for filtered events
    plt.xticks(index, events.astype(int), rotation=45)  # Show bars only for every event with the specified gap
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


# Chat examples
def plot_stability_over_time():
    plt.clf()
    plt.figure(figsize=(10, 6))

    # Your plotting code here
    events = np.arange(0, 100)
    rule1_stability = np.cumsum(np.random.exponential(0.1, 100))
    rule2_stability = np.cumsum(np.random.uniform(0, 1, 100))
    rule3_stability = np.cumsum(np.random.normal(0.5, 0.2, 100))

    plt.plot(events, rule1_stability, label='Static Rule', color='blue')
    plt.plot(events, rule2_stability, label='Dynamic Rule', color='red')
    plt.plot(events, rule3_stability, label='Smooth Rule', color='green')

    plt.title('Stability Analysis: Cumulative Updates Over Time')
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Number of Updates')
    plt.legend()
    plt.grid(True)

    # Save the plot instead of displaying it
    plt.savefig('stability_plot.png')
    plt.close()


def plot_community_size_impact():
    # Simulate data
    community_sizes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Different performance patterns for different rules
    rule1_perf = 0.8 - 0.3 * np.log(community_sizes / 10)  # Decreases with size
    rule2_perf = 0.6 + 0.2 * np.log(community_sizes / 10)  # Improves with size
    rule3_perf = 0.7 * np.ones_like(community_sizes)  # Stable across sizes

    # Add some noise
    np.random.seed(42)
    rule1_perf += np.random.normal(0, 0.05, len(community_sizes))
    rule2_perf += np.random.normal(0, 0.05, len(community_sizes))
    rule3_perf += np.random.normal(0, 0.05, len(community_sizes))

    plt.figure(figsize=(10, 6))
    plt.scatter(community_sizes, rule1_perf, label='Static Rule', color='blue')
    plt.scatter(community_sizes, rule2_perf, label='Dynamic Rule', color='red')
    plt.scatter(community_sizes, rule3_perf, label='Smooth Rule', color='green')

    # Add trend lines
    plt.plot(community_sizes, rule1_perf, '--', color='blue', alpha=0.5)
    plt.plot(community_sizes, rule2_perf, '--', color='red', alpha=0.5)
    plt.plot(community_sizes, rule3_perf, '--', color='green', alpha=0.5)

    plt.title('Rule Performance vs Community Size')
    plt.xlabel('Number of Agents in Community')
    plt.ylabel('Performance Score')
    plt.legend()
    plt.grid(True)

    # Save the plot instead of displaying it
    plt.savefig('Community Size Impact.png')
    plt.close()


def plot_agent_model_comparison():
    # Simulate data for different agent types and rules
    rules = ['Static Rule', 'Dynamic Rule', 'Smooth Rule']
    agent_types = ['Basic', 'Uniform', 'Gaussian']

    # Create performance data for each combination
    performance_data = np.array([
        [0.75, 0.65, 0.70],  # Basic agents
        [0.60, 0.80, 0.75],  # Uniform interval agents
        [0.65, 0.85, 0.80]  # Gaussian interval agents
    ])

    bar_width = 0.25
    r1 = np.arange(len(agent_types))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(12, 6))
    plt.bar(r1, performance_data[:, 0], width=bar_width, label='Static Rule', color='blue')
    plt.bar(r2, performance_data[:, 1], width=bar_width, label='Dynamic Rule', color='red')
    plt.bar(r3, performance_data[:, 2], width=bar_width, label='Smooth Rule', color='green')

    plt.title('Rule Performance Across Agent Models')
    plt.xlabel('Agent Type')
    plt.ylabel('Performance Score')
    plt.xticks([r + bar_width for r in range(len(agent_types))], agent_types)
    plt.legend()

    # Save the plot instead of displaying it
    plt.savefig('Agent Model Comparison.png')
    plt.close()


plot_agent_model_comparison()
plot_community_size_impact()
plot_stability_over_time()
