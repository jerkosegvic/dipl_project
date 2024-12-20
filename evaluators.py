from models import Model_LLM
from typing import List, Tuple, Union
import torch
from datasets import Boolq_dataset, MultiRC_dataset, ReCoRD_dataset, RACE_dataset

def evaluate_boolqe_example_llm(
        model: Model_LLM,
        example: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
    '''
    Evaluate a single BoolQ example and returns the 
    probability of the correct answer and if the model got it right
    '''
    input_ids, attention_mask, targets, input_ids_neg, attention_mask_neg, targets_neg = example
    loss, logits = model(input_ids, attention_mask, targets)
    loss_neg, logits_neg = model(input_ids_neg, attention_mask_neg, targets_neg)
    return loss < loss_neg

def evaluate_multirc_example_llm(
        model: Model_LLM,
        example: List[Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]],
        treshold: Union[float, None],
    ) -> Tuple[float, float, float]:
    '''
    Evaluate a single MultiRC example, returns accuracy, precission and recall
    '''
    results = []
    input_ids = torch.stack([x[1] for x in example])
    attention_mask = torch.stack(x[2] for x in example)
    targets = torch.stack(x[3] for x in example)

    results = sorted(results, key=lambda x: x[0])

    if not treshold:
        treshold = min(list(filter(lambda x: x[1])), key=lambda x: x[0])[0]
    
    TP = len(list(filter(lambda x: x[0] <= treshold and x[1])))
    FP = len(list(filter(lambda x: x[0] <= treshold and not x[1])))
    TN = len(list(filter(lambda x: x[0] > treshold and not x[1])))
    FN = len(list(filter(lambda x: x[0] > treshold and x[1])))
    return (TP + TN) / (TP + TN + FP + FN), TP / (TP + FP), TP / (TP + FN)

def evaluate_race_example_llm(
        model: Model_LLM,
        example: Tuple[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ) -> bool:
    '''
    Evaluate a single RACE example, returns if the example is correctly classfied
    '''
    correct, example = example
    input_ids = torch.stack([x[0] for x in example])
    attention_mask = torch.stack(x[1] for x in example)
    targets = torch.stack(x[2] for x in example)

    loss, logits = model(input_ids, attention_mask, targets)
    return torch.argmax(logits) == correct

def evaluate_record_example_llm(
        model: Model_LLM,
        example: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        top_k: int = 5
    ) -> Tuple[bool, int]:
    '''
    Evaluate a single ReCoRD example, returns the start and end of the answer
    '''
    ##TODO: implement this
    input_ids, attention_mask, targets = example[0]
    # find the first occurence of 1 in the targets
    ind = torch.argmax(targets)
    input_ids_q = input_ids[:ind]
    decodings = model.decode(input_ids_q, top_k)

    