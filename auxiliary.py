from typing import Tuple, Union, Literal, List, Dict
import re

class MultiRC_question:
    def __init__(
        self,
        passage_index: int,
        question: str,
        sentences_used: List[int],
        answers: Dict[Union[str, str | str,bool]]
    ):
        self.passage_index = passage_index
        self.question = question
        self.sentences_used = sentences_used
        self.answers = [a['text'] for a in answers]
        self.correct = [a['isAnswer'] for a in answers]

class ReCoRD_question:
    def __init__(
        self,
        passage_index: int,
        query: str,
        answer: str,
    ) -> None:
        self.passage_index = passage_index
        self.query = query
        self.answer = answer
        self.w_ind = [i for i, j in enumerate(query.split(" ")) if re.search(".*@placeholder.*", j) ][0]