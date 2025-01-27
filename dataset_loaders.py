from datasets import Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset
from auxiliary import MultiRC_question, ReCoRD_question
import os
import pandas as pd
import json
import re
from typing import Type, TypeVar
from transformers import AutoTokenizer
import pickle
from typing import Union, Tuple

PROC_DATA_DIR = 'processed_data'

TBQ = TypeVar('TBQ', bound=Boolq_dataset)
def load_boolq(
        path: str,
        tokenizer: AutoTokenizer,
        save_name: str = None,
        Dataset_: Type[TBQ] = Boolq_dataset,
        max_length: int = 1024
    ) -> TBQ:
    if save_name is None:
        save_name = "boolq-" + path.split('/')[-1].split('.')[0] + '.pkl'

    save_path = os.path.join(PROC_DATA_DIR, save_name)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
        
    with open(path) as f:
        raw_dataset = f.readlines()

    dataset = pd.DataFrame([json.loads(x) for x in raw_dataset])
    passages = dataset['passage'].tolist()
    questions = dataset['question'].tolist()
    answers = dataset['answer'].tolist()

    ds = Dataset_(passages, questions, answers, tokenizer, max_length)
    with open(save_path, 'wb') as f:
        pickle.dump(ds, f)
    
    return ds


TMR = TypeVar('TMR', bound=MultiRC_dataset)
def load_multirc(
        path: str,
        tokenizer: AutoTokenizer,
        tokenizer_rag: AutoTokenizer = None,
        save_name: str = None,
        Dataset_: Type[TMR] = MultiRC_dataset,
        max_length: int = 1024,
        max_length_rag: int = 512,
        ind_range: Union[Tuple[int, int], None] = None,
        positive_only: bool = False
    ) -> TMR:
    if save_name is None:
        name_appendix = ""
        if ind_range is not None:
            name_appendix = f"-{ind_range[0]}-{ind_range[1]}"

        if positive_only:
            name_appendix += "-pos_only"

        if Dataset_.class_name() == "multirc_base":
            save_name = "multirc-" + path.split('/')[-1].split('.')[0] + name_appendix + '.pkl'
        else:
            save_name = "multirc-" + Dataset_.class_name() + "-" + path.split('/')[-1].split('.')[0] + name_appendix + '.pkl'

    save_path = os.path.join(PROC_DATA_DIR, save_name)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    with open(path) as f:
        raw_dataset = json.load(f)

    dataset = pd.DataFrame(raw_dataset['data'])
    if ind_range is not None:
        dataset = dataset.iloc[ind_range[0]:ind_range[1]]
    
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
    
    if tokenizer_rag is not None:
        ds = Dataset_(paragraphs, questions_list, tokenizer, tokenizer_rag, max_length, max_length_rag, positive_only)
    else:
        ds = Dataset_(paragraphs, questions_list, tokenizer, max_length)
    with open(save_path, 'wb') as f:
        pickle.dump(ds, f)

    return ds

TRA = TypeVar('TRA', bound=RACE_dataset)
def load_race(
        path: str,
        tokenizer: AutoTokenizer,
        save_name: str = None,
        Dataset_: Type[TRA] = RACE_dataset,
        max_length: int = 1024
    ) -> TRA:
    if save_name is None:
        save_name = "race-" + path.split('/')[-1].split('.')[0] + '.pkl'

    save_path = os.path.join(PROC_DATA_DIR, save_name)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    
    dataset = pd.read_parquet(path)

    passages = []
    passage_inds = []
    questions = []
    answers = []
    correct_answer_inds = []
    id_ind = {}
    for (i,(id, article, question, options, answer)) in enumerate(dataset[['example_id', 'article', 'question', 'options','answer']].values):
        if id not in id_ind.keys():
            id_ind[id] = len(passages)
            passages.append(article)
            
        passage_inds.append(id_ind[id])
        questions.append(question)
        answers.append(options)
        correct_answer_inds.append(int(ord(answer) - ord('A')))
    
    ds = Dataset_(
        passages,
        passage_inds,
        questions,
        answers,
        correct_answer_inds,
        tokenizer,
        max_length=max_length
    )
    with open(save_path, 'wb') as f:
        pickle.dump(ds, f)

    return ds

TRC = TypeVar('TRC', bound=ReCoRD_dataset)
def load_record(
        path: str, 
        tokenizer: AutoTokenizer,
        save_name: str = None,
        Dataset_: Type[TRC] = ReCoRD_dataset,
        max_length: int = 1024
    ) -> TRC:
    if save_name is None:
        save_name = "record-" + path.split('/')[-1].split('.')[0] + '.pkl'

    save_path = os.path.join(PROC_DATA_DIR, save_name)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
        
    with open(path) as f:
        raw_dataset = json.load(f)
    
    dataset = pd.DataFrame(raw_dataset['data'])
    passages = []
    qs = []
    for (i,(passage, qas)) in enumerate(zip(dataset['passage'], dataset['qas'])):
        passages.append(passage['text'])
        for q in qas:
            question = ReCoRD_question(
                query=q['query'],
                answers=[a['text'] for a in q['answers']],
                answers_span=[(a['start'], a['end']) for a in q['answers']],
                paragraph_index=i
            )
            qs.append(question)
    
    ds = Dataset_(passages, qs, tokenizer, max_length)
    with open(save_path, 'wb') as f:
        pickle.dump(ds, f)
    
    return ds
