from typing import Tuple, Union, Literal, List, Dict
import re

class MultiRC_question:
    def __init__(
        self,
        passage_id,
        passage_index: int,
        question: str,
        sentences_used: List[int],
        answers: Dict[str, Union[str | bool]]
    ):
        self.passage_id = passage_id
        self.passage_index = passage_index
        self.question = question
        self.sentences_used = sentences_used
        self.answers = [a['text'] for a in answers]
        self.correct = [a['isAnswer'] for a in answers]

class ReCoRD_question:
    def __init__(
        self,
        query: str,
        answers: List[str],
        answers_span: List[Tuple[int, int]],
        paragraph_index: int
    ) -> None:
        self.paragraph_id = paragraph_index
        self.query = query
        self.answers = answers
        self.answers_span = answers_span
        self.w_ind = re.search("@placeholder", query).span()