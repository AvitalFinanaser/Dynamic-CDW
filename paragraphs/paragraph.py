class Paragraph:

    def __init__(self, text, paragraph_id, name):
        self._text = text
        self._paragraph_id = paragraph_id
        self._name = name

    def __eq__(self, other):
        if isinstance(other, Paragraph):
            return self._paragraph_id == other._paragraph_id and self._text == other._text and self._name == other.name
        return False

    @property
    def paragraph_id(self):
        # Getter for paragraph's id
        return self._paragraph_id

    @paragraph_id.setter
    def paragraph_id(self, value):
        # Setter for paragraph's id
        self._paragraph_id = value

    @property
    def name(self):
        # Getter for paragraph's name
        return self._name

    @name.setter
    def name(self, value):
        # Setter for paragraph's name
        self._name = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        # Setter for paragraph's text
        self._text = value

    @staticmethod
    def get_paragraph_name_by_id(paragraph_id, paragraphs):
        # Search for the paragraph in the provided list
        for paragraph in paragraphs:
            if paragraph.paragraph_id == paragraph_id:
                return paragraph.name
        return None


    def __str__(self) -> str:
        """
        Returns the id, name and text of paragraph
        """
        return f"ID: {self._paragraph_id}, Name: {self._name}, Text: {self._text}"


