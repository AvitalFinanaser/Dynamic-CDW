from simulation import SimulationConfig, AgentConfig, AgentType, ContextConfig, Scheduler
from rules import *
from simulation import solution_analysis as SA
from events import *


def aggregation_rule():
    threshold = 0.5
    CSF = ["APS", "NGS"]
    d_events = EventsDynamicProperty()
    d_paragraphs = ParagraphsDynamicProperty()
    Fexp_1 = ExpSmoothingFunction(alpha=0.1)
    Fexp_3 = ExpSmoothingFunction(alpha=0.3)
    Fexp_5 = ExpSmoothingFunction(alpha=0.5)
    rules = []

    # 1-SC: CSF: {APS, NGS}
    cS1 = StaticCondition(CSF=CSF[0], threshold=threshold)
    cS2 = StaticCondition(CSF=CSF[1], threshold=threshold)
    rules.extend([cS1, cS2])

    # 1-harshD: CSF: {APS, NGS}, property:{events, paragraphs}
    hD1 = HarshDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_events, t=50)
    hD2 = HarshDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_events, t=50)

    hD3 = HarshDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_paragraphs, t=50)
    hD4 = HarshDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_paragraphs, t=50)
    rules.extend([hD1, hD2, hD3, hD4])

    # 1-smoothD: CSF: {APS, NGS}, property:{events, paragraphs}, Functions:{Fexp_0.1, Fexp_0.3, Fexp_0.5}
    sD1 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_events, F=Fexp_1)
    sD2 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_events, F=Fexp_1)
    sD3 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_1)
    sD4 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_1)

    sD5 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_events, F=Fexp_3)
    sD6 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_events, F=Fexp_3)
    sD7 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_3)
    sD8 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_3)

    sD9 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_events, F=Fexp_5)
    sD10 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_events, F=Fexp_5)
    sD11 = SmoothDynamicCondition(CSF=CSF[0], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_5)
    sD12 = SmoothDynamicCondition(CSF=CSF[1], threshold=threshold, dynamic_property=d_paragraphs, F=Fexp_5)
    rules.extend([sD1, sD2, sD3, sD4, sD5, sD6, sD7, sD8, sD9, sD10, sD11, sD12])
    return rules


def test_solution(rules, E):
    for r in rules:
        print(f"\nSolution of rule: {r}\n")
        sol = r.solution_1Condition(E=E)
        print(f"solution for rule {r} \n {SA.S_E(sol)}")
        sols = SA.solutions(E=E, condition=r)
        SA.print_solutions_names(sol_list=sols)


def test_updates(rules, E):
    for r in rules:
        print(f"\nSolution of rule: {r}\n")
        print(SA.updates_and_solutions(E=E, condition=r))


def test_satisfaction_analysis(rule: Condition, E):
    """
    Test satisfaction analysis using a single aggregation rule.
    Shows basic usage of the three main satisfaction analysis methods.
    """
    # Test all three satisfaction analysis methods
    print(f"\nTesting Satisfaction Analysis Methods for rule {rule}:")
    # Get solution using the rule
    solution = rule.solution_1Condition(E)

    # 1. Calculate weighted preference matrix
    matrix = SA.weighted_preference_matrix(event_list=E, final_solution=solution)

    # 2. Print the matrix in formatted form
    SA.print_weighted_preference_matrix(weighted_matrix=matrix, event_list=E)

    # 3. Calculate statistics
    stats = SA.calculate_satisfaction_statistics(weighted_matrix=matrix)
    print(f"\nSummary Statistics:")
    print(stats)
    print(f"Average Satisfaction: {stats['average']:.3f}")
    print(f"Total Satisfaction: {stats['sum']:.3f}")
    print(f"Min Satisfaction: {stats['min']:.3f}")
    print(f"Max Satisfaction: {stats['max']:.3f}")



def generate_static_rules():
    """
    Generates rules for CDW system experiments.
    1. static - CSF: ["APS", "AM", "APS_r"]
    2. parma (dynamic property) - events, paragraphs
    3. alpha (smoothing parameter) - (0.05,0.1,0.3,0.5, 0.7,1)
    """
    # Rules options
    threshold = 0.5
    CSF = ["APS", "APS_r", "AM"]
    rules = []

    # 1-SC: CSF: {APS, "AM", "APS_r"}
    cS1 = StaticCondition(CSF="APS", threshold=threshold)
    cS2 = StaticCondition(CSF="APS_r", threshold=threshold)
    cS3 = StaticCondition(CSF="AM", threshold=threshold, beta=0.05)
    cS4 = StaticCondition(CSF="AM", threshold=threshold, beta=0.1)

    rules.extend([cS1, cS2, cS3, cS4])
    return rules


if __name__ == "__main__":
    # Test Case 1: Unstructured Agents
    config = SimulationConfig(
        num_agents=3,
        num_events=20,
        num_lists=2,
        agent_config=AgentConfig(agent_type=AgentType.UNSTRUCTURED),
        context_config=ContextConfig(context_type="synthetic", num_paragraphs=100),
        random_seed=10
    )
    scheduler = Scheduler(config)

    # Generate single event list
    community, event_list = scheduler.schedule_single_instance()
    # Print and validate event list
    print(f"Single Event List for Configuration {config}:")
    print(event_list.events_df())


    # Tests
    #rules = aggregation_rule()
    # test_solution(rules=rules, E=event_list)
    # test_updates(rules=rules, E=event_list)
    #test_satisfaction_analysis(rule=rules[0], E=event_list)
    # Test of new static rules
    rules = generate_static_rules()
    for rule in rules:
        test_satisfaction_analysis(rule=rule, E=event_list)
