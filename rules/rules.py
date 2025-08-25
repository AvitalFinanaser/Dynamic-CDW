from abc import ABC, abstractmethod
from paragraphs import *
from events import *
import math


# Static-CSF
def APS(p: Paragraph, E: EventList) -> float:
    """
    :param p: Paragraph
    :param E: Event list
    :return: APS consensus score for p given E
    """
    # Approval Proportion Score
    score = 0.0
    total_votes = E.p_plus(p) + E.p_minus(p)
    if p in E.P() and E.events and total_votes != 0:  # Input check: p in P(E), else returns 0
        score = E.p_plus(p) / total_votes
    return score


def APS_r(p: Paragraph, E: EventList) -> float:
    """
    APS_r:= p^+_r / p^+_r + p^-_r
    :param p: Paragraph
    :param E: Event list
    :return: APS_r consensus score for p given E
    """
    # Approval Proportion Score
    score = 0.0
    p_plus_r = E.p_plus_r(p=p)
    p_minus_r = E.p_minus_r(p=p)
    total_relative = p_plus_r + p_minus_r
    if p in E.P() and E.events and total_relative != 0:  # Input check: p in P(E), else returns 0
        score = p_plus_r / total_relative
    return score


def AM(p: Paragraph, E: EventList, beta: float, actives: int = 0) -> float:
    """
    AM - beta absolute majority. AM := APS_r if p^+ >= beta * |A|, else, 0.
    :param actives: number of active agents
    :param p: Paragraph
    :param E: Event list
    :param beta: proportion of absolute majority
    :return: AM consensus score for p given E
    """
    # Approval Proportion Score
    score = 0.0
    p_plus = E.p_plus(p)
    # Number of actives agents was not supplied
    if actives == 0:
        actives = E.active_Na()
    if p_plus >= beta * actives:
        score = APS_r(p, E)
    return score


def NGS(p: Paragraph, E: EventList) -> float:
    """
    :param p: Paragraph
    :param E: Event list
    :return: NGS consensus score for p given E
    """
    # Approval Proportion Score
    score = 0.0
    if p in E.P() and E.events and len(E.A_E()) != 0:  # Input check: p in P(E), else returns 0
        gap = E.p_plus(p) - E.p_minus(p)
        num_events = len(E.events)
        score = 0.5 * (1 + gap / num_events)
    return score


def getStaticCSF(CSF: str, E: EventList, p: Paragraph, beta: float = 0, actives: int = 0) -> float:
    """
    :param actives:
    :param beta: proportion of absolute majority
    :param CSF: Consensus scoring function
    :param E: Event list Object
    :param p: Paragraph
    :return: Static Consensus score of p in E
    """
    score = 0.0
    if CSF == "APS":
        score = APS(p=p, E=E)
    elif CSF == "NGS":
        score = NGS(p=p, E=E)
    elif CSF == "APS_r":
        score = APS_r(p=p, E=E)
    elif CSF == "AM":
        score = AM(p=p, E=E, beta=beta, actives=actives)
    return score


# Smoothing functions - for smoothD


class SmoothingFunction(ABC):
    def __init__(self, alpha: float):
        """
        Smoothing function used in smoothD.
        :param alpha: Smoothing parameter
        """
        self.alpha = alpha

    @abstractmethod
    def calculate(self, x: float, t: int):
        pass

    @abstractmethod
    def __str__(self):
        pass


class ExpSmoothingFunction(SmoothingFunction):
    def __init__(self, alpha: float):
        """
        Exponential Smoothing function.
        :param alpha: Smoothing parameter
        """
        super().__init__(alpha)

    def calculate(self, x: float, t: int):
        return x * math.exp(-t * self.alpha * (1 - x))

    def __str__(self):
        """
        Returns a string with the function type and alpha parameter.
        """
        return f"Function=exp_alpha={self.alpha}"


class LinearSmoothingFunction(SmoothingFunction):
    def __init__(self, alpha: float):
        """
        Exponential Smoothing function.
        :param alpha: Smoothing parameter
        """
        super().__init__(alpha)

    def calculate(self, x: float, t: int):
        return x * (1 - self.alpha * (t / (t + 1)))

    def __str__(self):
        """
        Returns a string with the function type and alpha parameter.
        """
        return f"Function=linear_alpha={self.alpha}"


# Dynamic properties - for dynamic-CSF


class DynamicProperty(ABC):
    """
    An abstract class for the system's dynamic property
    """

    @abstractmethod
    def dynamicIndexes(self, E: EventList) -> list:
        """
        Returns a list of event indices where the dynamic property changes.
        :param E: Event list object.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns the name of the dynamic property.
        """
        pass

    @abstractmethod
    def countDynamics(self, E: EventList) -> int:
        """
        Returns the count of the dynamic property in the event list.
        :param E: Event list object.
        """
        pass

    def tIndex(self, eventlist: EventList, event: int) -> float:
        """
        :param eventlist: Event list
        :param event: An event index given event list E
        :return: t value given event i
        """
        pass

    def CSHarshScore(self, t: int, beta: float, p: Paragraph, E: EventList, static_CSF: str) -> float:
        """
        :param beta: parameter of absolute majority
        :param static_CSF: A static consensus scoring function
        :param t: Number of events limitation
        :param p: Paragraph
        :param E: Event list
        :return: Dynamic consensus score for p given E subtract by t events
        """
        pass

    def CSSmoothScore(self, CSF: str, beta: float, E: EventList, p: Paragraph, t: int, F: SmoothingFunction) -> float:
        """
        :param beta: parameter of absolute majority
        :param CSF: Static Consensus scoring function
        :param E: Event list Object
        :param p: Paragraph
        :param t: Current value of the dynamizer parameter
        :param alpha: Smoothing
        :param F:  function
        :return: SmoothD Consensus score of p in E
        """
        pass


# Property - events in event list E
class EventsDynamicProperty(DynamicProperty):

    def dynamicIndexes(self, E: EventList) -> list:
        # For events, the property changes at every event.
        return list(range(len(E.events)))

    def __str__(self) -> str:
        return "events"

    def countDynamics(self, E: EventList) -> int:
        # The dynamic property is the number of events.
        return len(E.events)

    def tIndex(self, eventlist: EventList, event: int) -> float:
        """
        :param eventlist: Event list
        :param event: An event index given event list E
        :return: t value given event i
        """
        return event

    def CSHarshScore(self, t: int, beta: float, p: Paragraph, E: EventList, static_CSF: str, actives: int = 0) -> float:
        """
        :param actives:
        :param beta: parameter of absolute majority
        :param static_CSF: A static consensus scoring function
        :param t: Number of events limitation
        :param p: Paragraph
        :param E: Event list
        :return: Dynamic consensus score for p given E subtract by t events
        """
        if E.events and len(E.events) <= t:  # There are less than t events in E
            e_calc = E
        elif E.events:  # There are more than t events in E
            temp = E.E_i(t)  # List of first t events in E
            e_calc = EventList()
            e_calc.events = temp
        else:  # The event list is empy
            print("Invalid input to t_events_Dynamizer - event list is empty")
            return 0
        return getStaticCSF(CSF=static_CSF, E=e_calc, p=p, beta=beta, actives=actives)

    def CSSmoothScore(self, CSF: str, beta: float, E: EventList, p: Paragraph, t: int, F: SmoothingFunction,
                      actives: int = 0) -> float:
        """
        :param actives:
        :param beta: parameter of absolute majority
        :param CSF: Static Consensus scoring function
        :param E: Event list Object
        :param p: Paragraph
        :param t: Current value of the dynamizer parameter
        :param F:  function
        :return: SmoothD Consensus score of p in E
        """
        # Subtract the event list to t events
        if not E.events:
            print("Invalid input - event list is empty")
            return 0
        score = getStaticCSF(CSF=CSF, E=E, p=p, beta=beta, actives=actives)
        # According to smoothing function - dynamic-CSF
        return F.calculate(x=score, t=t)


# Property - paragraphs suggested P(E)

class ParagraphsDynamicProperty(DynamicProperty):

    def dynamicIndexes(self, E: EventList) -> list:
        """
        Track the events where new paragraphs are suggested (E.P() changes).
        :param E: Event list object.
        :return: List of event indices where new paragraphs are suggested.
        """
        indexes = []
        total_paragraphs = []

        for i, event in enumerate(E.events):
            p = event.paragraph
            if p not in total_paragraphs:
                indexes.append(i)  # Add the event index where the paragraphs change
                total_paragraphs.append(p)

        return indexes

    def __str__(self) -> str:
        return "paragraphs"

    def countDynamics(self, E: EventList) -> int:
        # Count the number of paragraphs suggested
        return len(E.P())

    def tIndex(self, eventlist: EventList, event: int) -> float:
        """
        :param eventlist: Event list
        :param event: An event index given event list E
        :return: t value given event i
        """
        indexes = self.dynamicIndexes(E=eventlist)
        # Iterate through the change indexes and count how many have occurred by the given event index
        t_value = 0
        for idx in indexes:
            if idx <= event:
                t_value += 1
            else:
                break  # Stop counting once we've passed the event index

        return t_value

    def CSHarshScore(self, t: int, beta: float, p: Paragraph, E: EventList, static_CSF: str, actives: int = 0) -> float:
        """
        :param beta: absolute majority
        :param static_CSF: A static consensus scoring function
        :param t: Number of events limitation
        :param p: Paragraph
        :param E: Event list
        :return: Dynamic consensus score for p given E subtract by t paragraphs suggested
        """
        if not E.events:
            print("Invalid input to t_paragraphs_Dynamizer - event list is empty")
            return 0

        if len(E.P()) <= t:  # There are less than t paragraphs suggested in E
            e_calc = E
        else:  # There are more than t paragraphs suggested in E (none empty list)
            e_calc = EventList()
            total_paragraphs = []
            for i, event in enumerate(E.events):
                p = event.paragraph
                if p not in total_paragraphs:
                    total_paragraphs.append(p)

                if len(total_paragraphs) >= t:
                    e_calc.events = E.E_i(t)
                    break

        return getStaticCSF(CSF=static_CSF, E=e_calc, p=p, beta=beta, actives=actives)

    def CSSmoothScore(self, CSF: str, beta: float, E: EventList, p: Paragraph, t: int, F: SmoothingFunction) -> float:
        """
        :param beta: absolute majority
        :param CSF: Static Consensus scoring function
        :param E: Event list Object
        :param p: Paragraph
        :param t: Current value of the dynamizer parameter
        :param F:  function
        :return: SmoothD Consensus score of p in E
        """
        if not E.events:
            print("Invalid input - event list is empty")
            return 0
        score = self.CSHarshScore(static_CSF=CSF, t=t, E=E, p=p, beta=beta)
        # According to smoothing function - dynamic-CSF
        return F.calculate(x=score, t=t)


def getHarshCSF(CSF: str, beta: float, E: EventList, p: Paragraph, t: int, dynamic_property: DynamicProperty) -> float:
    """
    :param beta: absolute majority
    :param CSF: Static Consensus scoring function
    :param E: Event list Object
    :param p: Paragraph
    :param t: Limitation value of the dynamizer parameter
    :param dynamic_property: Dynamic parameter to dynamize
    :return: HarshD Consensus score of p in E
    """
    return dynamic_property.CSHarshScore(t=t, p=p, E=E, static_CSF=CSF, beta=beta)


# Smoothing functions
def FExp(alpha: float, x: float, t: int):
    return x * math.exp(-t * alpha * (1 - x))


def getSmoothCSF(CSF: str, E: EventList, beta: float, p: Paragraph, t: int, dynamic_property: DynamicProperty,
                 F: SmoothingFunction):
    """
    :param beta: absolute majority
    :param CSF: Static Consensus scoring function
    :param E: Event list Object
    :param p: Paragraph
    :param t: Current value of the dynamizer parameter
    :param dynamic_property: Dynamizer parameter
    :param F: Smoothing function
    :return: SmoothD Consensus score of p in E
    """
    '''
    e_calc = E
    # Retrieve static-CSF according to sub list of event list according to dynamic property
    if dynamic_property == "events":
        temp = EventList()
        temp.events = E.E_i(t)
        e_calc = temp
    score = getStaticCSF(CSF=CSF, E=e_calc, p=p)
    # According to smoothing function - dynamic-CSF
    if F == "exp":
        return FExp(alpha=alpha, x=score, t=t)
    return -1.0
    '''
    return dynamic_property.CSSmoothScore(CSF=CSF, E=E, p=p, t=t, F=F, beta=beta)


# Base class for conditions
class Condition(ABC):
    def __init__(self, CSF: str, threshold: float, beta: float = 0):
        """
        Conditioned based aggregation rule.
        :param CSF: Consensus scoring function
        :param threshold: Value between 0 and 1
        """
        self.CSF = CSF
        self.threshold = threshold
        self.beta = beta

    @abstractmethod
    def evaluate(self, E, p, t=None, actives: int = 0):
        pass

    @abstractmethod
    def compliance(self, E: EventList, p: Paragraph, t=None, actives: int = 0):
        pass

    def solution_1Condition(self, E: EventList) -> list:
        """
         Solution to 1-Condition rule
        :param E: Event list Object
        :return: Solution as the list of paragraphs to be included in the document
        """
        pass

    def __str__(self):
        """
        Returns a string with the attributes of the class.
        """
        return f"Condition(CSF={self.CSF}, threshold={self.threshold})"


class StaticCondition(Condition):
    def __init__(self, CSF: str, threshold: float, beta: float = 0):
        super().__init__(CSF, threshold, beta)

    def evaluate(self, E, p, t=None, actives: int = 0) -> float:
        # Call getStaticCSF to evaluate the paragraph using the CSF string
        return getStaticCSF(CSF=self.CSF, E=E, p=p, beta=self.beta, actives=actives)

    def compliance(self, E: EventList, p: Paragraph, t=None, actives: int = 0):
        return self.evaluate(E=E, p=p, actives=actives) >= self.threshold

    def solution_1Condition(self, E: EventList) -> list:
        """
         Solution to 1- static condition rule (1-SC)
        :param E: Event list Object
        :return: Solution as the list of paragraphs to be included in the document
        """
        solution = []
        actives = 0
        if self.CSF == "AM":
            actives = E.active_Na()
        for p in E.P():  # For every paragraph in P(E)
            include = self.compliance(E=E, p=p, actives=actives)
            # Include if condition holds, else remove
            if p in solution and include == False:
                solution.remove(p)
            elif p not in solution and include == True:
                solution.append(p)
        return solution

    def __str__(self):
        """
        Returns a string with the attributes of the class.
        """
        beta_str = f", beta={self.beta}" if hasattr(self, 'beta') and self.beta > 0 else ""
        return f"(CSF={self.CSF}{beta_str}, threshold={self.threshold})"


# Harsh Dynamic Condition class (t and dynamic property)

class HarshDynamicCondition(Condition):
    def __init__(self, CSF: str, threshold: float, dynamic_property: DynamicProperty, t: int, beta: float = 0):
        super().__init__(CSF, threshold, beta)
        self.dynamic_property = dynamic_property
        self.t = t

    def evaluate(self, E, p, t=None, actives: int = 0) -> float:
        # Logic for evaluating harsh dynamic CSF
        return getHarshCSF(CSF=self.CSF, t=self.t, dynamic_property=self.dynamic_property, E=E, p=p, beta=self.beta)

    def compliance(self, E: EventList, p: Paragraph, t=None, actives: int = 0):
        return self.evaluate(E=E, p=p) >= self.threshold

    def solution_1Condition(self, E: EventList) -> list:
        """
         Solution to 1- static condition rule (1-SC)
        :param E: Event list Object
        :return: Solution as the list of paragraphs to be included in the document
        """
        solution = []
        actives = 0
        if self.CSF == "AM":
            actives = E.active_Na()
        for p in E.P():  # For every paragraph in P(E)
            include = self.compliance(E=E, p=p, actives=actives)
            # Include if condition holds, else remove
            if p in solution and include == False:
                solution.remove(p)
            elif p not in solution and include == True:
                solution.append(p)
        return solution

    def __str__(self):
        """
        Returns a string with the attributes of the class.
        """
        beta_str = f", beta={self.beta}" if hasattr(self, 'beta') and self.beta > 0 else ""
        return f"(CSF={self.t}_{self.dynamic_property}, {self.CSF}{beta_str}, threshold={self.threshold})"


# Smooth Dynamic Condition class (alpha, F, and dynamic property)
class SmoothDynamicCondition(Condition):
    def __init__(self, CSF: str, threshold: float, dynamic_property: DynamicProperty, F: SmoothingFunction,
                 beta: float = 0):
        super().__init__(CSF, threshold, beta)
        self.dynamic_property = dynamic_property
        self.F = F

    def evaluate(self, E, p, t=None, actives: int = 0):
        # Logic for evaluating smooth dynamic CSF
        return getSmoothCSF(CSF=self.CSF, dynamic_property=self.dynamic_property, F=self.F, E=E, p=p, t=t,
                            beta=self.beta)

    def compliance(self, E: EventList, p: Paragraph, t=None, actives: int = 0):
        return self.evaluate(E=E, p=p, t=t) >= self.threshold

    def solution_1Condition(self, E: EventList) -> list:
        """
         Solution to 1-Condition rule
        :param E: Event list Object
        :return: Solution as the list of paragraphs to be included in the document
        """

        '''
        solution = []

        if print(self.dynamic_property) == "events":

            for t, event in enumerate(E.events, start=1):

                # Sublist of the first t events
                E_t = EventList()
                E_t.events = E.E_i(t)
                for p in E_t.P():
                    include = self.compliance(E=E_t, p=p, t=t)
                    # Include if condition holds, else remove
                    if p in solution and include == False:
                        solution.remove(p)
                    elif p not in solution and include == True:
                        solution.append(p)

        '''
        solution = []
        t = 0

        # indexes of events (starting with 0)
        indexes = list(range(len(E.events)))

        # For any index where the static scoring function can change
        for i in indexes:

            # t value given dynamic property
            t = self.dynamic_property.tIndex(eventlist=E, event=i)

            # Sublist of i+1 events - E_t
            event_count = i + 1
            E_t = EventList()
            E_t.events = E.E_i(event_count)
            actives = 0
            if self.CSF == "AM":
                actives = E.active_Na()
            for p in E_t.P():
                include = self.compliance(E=E_t, p=p, t=t, actives=actives)
                # Include if condition holds, else remove
                if p in solution and include == False:
                    solution.remove(p)
                elif p not in solution and include == True:
                    solution.append(p)
        return solution

    def __str__(self):
        """
        Returns a string with the attributes of the class.
        """
        beta_str = f"_beta={self.beta}" if hasattr(self, 'beta') and self.beta > 0 else ""
        return f"(CSF={self.CSF}{beta_str}, {self.F}_{self.dynamic_property}, threshold={self.threshold})"
