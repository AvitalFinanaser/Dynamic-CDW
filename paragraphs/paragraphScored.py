from paragraphs.paragraph import Paragraph


class ParagraphWithScore(Paragraph):
    """
    Represents a paragraph with a score between 0 and 1.
    New Attributes:
        _score (float): The score of the paragraph.
    """

    def __init__(self, text, paragraph_id, name, score=0.0):
        """
        Initialize a ParagraphWithScore instance - inheritance  of Paragraph
        :param text: The text of the paragraph.
        :param score: The score of the paragraph in [0,1], default=0
        """
        super().__init__(text, paragraph_id, name)
        self._score = score

    def __eq__(self, other):
        """
        Equality check for ParagraphWithScore instances.
        :param other: Another ParagraphWithScore instance
        :return: True if equal, False otherwise
        """
        if isinstance(other, ParagraphWithScore):
            return super().__eq__(other) and self._score == other._score
        return False

    @property
    def score(self):
        """
        Getter for score.
        :return: The score of the paragraph
        """
        return self._score

    @score.setter
    def score(self, value):
        """
        Setter for score.
        :param value: The new value for the paragraph's score
        """
        self._score = value


    def __str__(self) -> str:
        """
        Returns the id, name and text of paragraph
        """
        return f"ID: {self._paragraph_id}, Name: {self._name}, Text: {self._text}, Score: {self._score}"
