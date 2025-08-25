from agents import *
from paragraphs import *


class Event:
    def __init__(self, a: 'Agent', p: 'Paragraph', v: int):
        self._agent = a
        self._paragraph = p
        self._vote = v

    def __str__(self):
        return f"({self._agent.name}, {self._paragraph.name}, {self._vote})"

    @property
    def agent(self):
        # Getter for agent's id
        return self._agent

    @property
    def paragraph(self):
        # Getter for agent's id
        return self._paragraph

    @property
    def vote(self):
        # Getter for agent's id
        return self._vote

    @agent.setter
    def agent(self, value):
        self._agent = value

    @paragraph.setter
    def paragraph(self, value):
        self._paragraph = value

    @vote.setter
    def vote(self, value):
        self._vote = value

