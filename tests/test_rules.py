from rules import *


def generate_smooth_rules():
    """
    Generates rules for CDW system experiments.
    1. static - CSF: ["APS", "AM", "APS_r"]
    2. parma (dynamic property) - events, paragraphs
    3. alpha (smoothing parameter) - (0.05,0.1,0.3,0.5, 0.7,1)
    """
    # Rules options
    threshold = 0.5
    CSFs = ["APS", "APS_r", "AM"]
    alphas = [0.05, 0.1, 0.3, 0.5, 0.7, 1]
    betas = [0.05, 0.1]
    dynamic_properties = [
        EventsDynamicProperty(),
        ParagraphsDynamicProperty()
    ]
    # Generate smoothing functions for each alpha
    smoothing_functions = {alpha: ExpSmoothingFunction(alpha=alpha) for alpha in alphas}
    rules = []

    # Generate rules
    for CSF in CSFs:
        for dynamic_property in dynamic_properties:
            for alpha, F in smoothing_functions.items():
                # For CSF other than "AM", add standard rules
                if CSF != "AM":
                    rule = SmoothDynamicConditaion(
                        CSF=CSF,
                        threshold=threshold,
                        dynamic_property=dynamic_property,
                        F=F
                    )
                    rules.append(rule)
                else:
                    # For CSF="AM", add rules with beta
                    for beta in betas:
                        rule = SmoothDynamicCondition(
                            CSF=CSF,
                            threshold=threshold,
                            dynamic_property=dynamic_property,
                            F=F,
                            beta=beta
                        )
                        rules.append(rule)
    return rules


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


def generate_harsh_rules():
    """
    Generates rules for CDW system experiments.
    1. static - CSF: ["APS", "AM", "APS_r"]
    2. parma (dynamic property) - events, paragraphs
    """
    # Rules options
    threshold = 0.5
    CSF = ["APS", "APS_r", "AM"]
    d_events = EventsDynamicProperty()
    d_paragraphs = ParagraphsDynamicProperty()
    rules = []

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


if __name__ == "__main__":

    # static rules
    print("\nStatic rules\n")
    rules = generate_static_rules()
    # print names
    for i in rules:
        print(i)

    # harsh rules
    print("\nHarsh rules\n")
    rules = generate_harsh_rules()
    for i in rules:
        print(i)

    # smooth rules
    print("\nSmooth rules\n")
    rules = generate_smooth_rules()
    for i in rules:
        print(i)

