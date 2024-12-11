from datasets import Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset
from auxiliary import MultiRC_question, ReCoRD_question
import os
import pandas as pd
import json
import re
from typing import Type, TypeVar
from transformers import AutoTokenizer


TBQ = TypeVar('TBQ', bound=Boolq_dataset)
def load_boolq(
        path: str,
        tokenizer: AutoTokenizer,
        Dataset_: Type[TBQ] = Boolq_dataset
    ) -> TBQ:
    with open(path) as f:
        raw_dataset = f.readlines()

    dataset = pd.DataFrame([json.loads(x) for x in raw_dataset])
    passages = dataset['passage'].tolist()
    questions = dataset['question'].tolist()
    answers = dataset['answer'].tolist()

    return Dataset_(passages, questions, answers, tokenizer)

TMR = TypeVar('TMR', bound=MultiRC_dataset)
def load_multirc(
        path: str,
        tokenizer: AutoTokenizer,
        Dataset_: Type[TMR] = MultiRC_dataset
    ) -> TMR:
    with open(path) as f:
        raw_dataset = json.load(f)

    dataset = pd.DataFrame(raw_dataset['data'])
    paragraphs = []
    questions_list = []
    for (i, (paragraph, id)) in enumerate(zip(dataset['paragraph'], dataset['id'])):
        questions = [MultiRC_question(
            passage_id=id,
            passage_index=i,
            question=q['question'],
            sentences_used=q['sentences_used'],
            answers=q['answers']
        ) for q in paragraph['questions']]
        questions_list.extend(questions)
        pattern = r"<b>Sent \d+: </b>(.*?)<br>"
        sentences = re.findall(pattern, paragraph['text'])
        paragraphs.append(sentences)
    
    return Dataset_(paragraphs, questions_list, tokenizer)

TRA = TypeVar('TRA', bound=RACE_dataset)
def load_race(
        path: str,
        tokenizer: AutoTokenizer,
        Dataset_: Type[TRA] = RACE_dataset
    ) -> TRA:
    dataset = pd.read_parquet(path)

    passages = []
    passage_inds = []
    questions = []
    answers = []
    correct_answer_inds = []
    id_ind = {}
    for (i,(id, article, question, options, answer)) in enumerate(dataset[['id', 'article', 'question', 'options','answer']].values):
        if id not in id_ind.keys():
            id_ind[id] = len(passages)
            passages.append(article)
            
        passage_inds.append(id_ind[id])
        questions.append(question)
        answers.append(options)
        correct_answer_inds.append(int(ord(answer) - ord('A')))
    
    return Dataset_(
        passages,
        passage_inds,
        questions,
        answers,
        correct_answer_inds,
        tokenizer
    )

TRC = TypeVar('TRC', bound=ReCoRD_dataset)
def load_record(
        path: str, 
        tokenizer: AutoTokenizer,
        Dataset_: Type[TRC] = ReCoRD_dataset
    ) -> TRC:
    with open(path) as f:
        raw_dataset = json.load(f)
    
    dataset = pd.DataFrame(raw_dataset['data'])
    passages = []
    qs = []
    for (passage, qas) in zip(dataset['passage'], dataset['qas']):
        guestion = ReCoRD_question(
            query=qas['query'],
            answers=[a['text'] for a in qas['answers']],
            answers_span=[(a['start'], a['end']) for a in qas['answers']]
        )
        qs.append(guestion)
        passages.append(passage)

    return Dataset_(passages, qs, tokenizer)
