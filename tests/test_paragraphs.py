from paragraphs import *


def test_paragraph_methods():
    # initialization
    p1 = Paragraph(paragraph_id=1, name="p1", text="Hi")
    print(f"Paragraph {p1} initialized")
    p2 = Paragraph(paragraph_id=1, name="p1", text="H")
    # equal
    print(f"Paragraphs {p1} &\n{p2} are identical ?\n{p1.__eq__(p2)}")
    # getters
    print(f"ID: {p1.paragraph_id}, Name: {p1.name}, Text: {p1.text}")
    # setters
    p1.text = "Bye"
    print(f"p1 now changed - {p1}")


def test_paragraphScored_methods():
    # initialization - with score
    p1 = ParagraphWithScore(paragraph_id=1, name="p1", text="Hi", score=0.5)
    print(f"Paragraph {p1} initialized")
    # initialization - without score
    p2 = ParagraphWithScore(paragraph_id=2, name="p1", text="H")
    print(f"Paragraph {p2} initialized")

    # equal
    print(f"Paragraphs {p1} &\n{p2} are identical ?\n{p1.__eq__(p2)}")
    # getters
    print(f"ID: {p1.paragraph_id}, Name: {p1.name}, Text: {p1.text}")
    # setters
    p1.text = "Bye"
    print(f"p1 now changed - {p1}")



if __name__ == "__main__":
    test_paragraph_methods()
    test_paragraphScored_methods()

