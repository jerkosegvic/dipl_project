from models import Model_LLM, Retriver
from typing import List, Tuple, Union
import torch
from datasets import Boolq_dataset, MultiRC_dataset, ReCoRD_dataset, RACE_dataset, RAG_MultiRC_dataset, RAG_MultiRC_dataset_TL
from typing import TypeVar
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import numpy as np

def auxilliary_calculate(
        results: List[Tuple[torch.Tensor, bool]], 
        return_details: bool = False,
        threshold: Union[float, None] = None,
        threshold_total: float = 0.75,
        loss: bool = True
    ) -> Tuple[float, float, float]:
    '''
    Auxilliary function to calculate accuracy, precision and recall
    '''
    if not loss:
        results = list(map(lambda x: (-x[0], x[1]), results))
    if not threshold:
        threshold = min(list(filter(lambda x: x[1], results)), key=lambda x: x[0])[0]
    
    TP = len(list(filter(lambda x: x[0] <= threshold and x[1], results)))
    FP = len(list(filter(lambda x: x[0] <= threshold and not x[1], results)))
    TN = len(list(filter(lambda x: x[0] > threshold and not x[1], results)))
    FN = len(list(filter(lambda x: x[0] > threshold and x[1], results)))
    
    if return_details:
        print(results)
        return TP, FP, TN, FN
    else:
        return ( (TP + TN) / (TP + TN + FP + FN) ) >= threshold_total

def evaluate_boolqe_example_llm(
        model: Model_LLM,
        example: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> torch.Tensor:
    '''
    Evaluate a single BoolQ example and returns the 
    probability of the correct answer and if the model got it right
    '''
    input_ids, attention_mask, targets, input_ids_neg, attention_mask_neg, targets_neg = example
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    targets = targets.to(device)
    input_ids_neg = input_ids_neg.to(device)
    attention_mask_neg = attention_mask_neg.to(device)
    targets_neg = targets_neg.to(device)
    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask, targets)
        loss_neg, logits_neg = model(input_ids_neg, attention_mask_neg, targets_neg)
    return loss.item() < loss_neg.item()

def evaluate_multirc_example_llm(
        model: Model_LLM,
        example: List[Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]],
        threshold: Union[float, None] = None,
        threshold_total: float = 0.75,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        return_details: bool = False,
        metric_func: callable = auxilliary_calculate
    ) -> Tuple[float, float, float]:
    '''
    Evaluate a single MultiRC example, returns accuracy, precission and recall
    '''
    ##TODO: FIX THIS
    corrects = [x[0] for x in example]
    input_ids = [x[1] for x in example]
    attention_masks = [x[2] for x in example]
    targets = [x[3] for x in example]
    results = []
    for (c,i,a,t) in zip(corrects, input_ids, attention_masks, targets):
        with torch.no_grad():
            res = model(i.to(device), a.to(device), t.to(device))[0].to('cpu')
        results.append((res, c))
        i.to('cpu')
        a.to('cpu')
        t.to('cpu')

    results = sorted(results, key=lambda x: x[0])

    return metric_func(results, return_details, threshold, threshold_total)
    
    
def evaluate_race_example_llm(
        model: Model_LLM,
        example: Tuple[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> bool:
    '''
    Evaluate a single RACE example, returns if the example is correctly classfied
    '''
    correct, example = example
    input_ids = [x[0].to(device) for x in example]
    attention_mask = [x[1].to(device) for x in example]
    targets = [x[2].to(device) for x in example]
    losses = []

    for (i,a,t) in zip(input_ids, attention_mask, targets):
        with torch.no_grad():
            loss, logits = model(i, a, t)
        losses.append(loss.item())

    return correct == torch.argmax(torch.tensor(losses))

def evaluate_record_example_llm(
        model: Model_LLM,
        example: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        top_k: int = 5
    ) -> Tuple[bool, int]:
    '''
    Evaluate a single ReCoRD example, returns the start and end of the answer
    '''
    ##TODO: implement this
    input_ids, attention_mask, targets = example[0]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    targets = targets.to(device)
    # find the first occurence of 1 in the targets
    ind = torch.argmax(targets)
    input_ids_q = input_ids[:ind]
    decodings = model.decode(input_ids_q, top_k)

def evaluate_multirc_example_retriver(
        model: Retriver,
        example: Tuple[List[Tuple[bool, dict, torch.Tensor]], Tuple[dict, torch.Tensor]],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        threshold: Union[float, None] = None,
        threshold_total: float = 0.75
    ) -> bool:
    '''
    Evaluate a single MultiRC example, returns if the example is correctly classfied
    '''
    docs, question = example
    results = []
    with torch.no_grad():
        q_enc = model.forward(question[0], question=True)
    for (cor,doc,_) in docs:
        with torch.no_grad():
            doc_enc = model.forward(doc, question=False)
            res = model.compare(doc_enc, q_enc)
        results.append((res, cor))

    return auxilliary_calculate(results, threshold=threshold, threshold_total=threshold_total, loss=False)


def evaluate_task_llm(
        model: Model_LLM,
        dataset: Union[Boolq_dataset, RACE_dataset, MultiRC_dataset, ReCoRD_dataset],
        device: torch.device,
        function: Union[
            evaluate_boolqe_example_llm,
            evaluate_multirc_example_llm,
            evaluate_race_example_llm,
            evaluate_record_example_llm
        ],
        evaluation_size: int = None
) -> float:
    '''
    Evaluate a Language Model on a dataset
    '''
    #model.to(device)
    correct = 0
    total = 0
    if not evaluation_size:
        evaluation_size = dataset.len_eval()

    model.eval()
    for ind in range(min(dataset.len_eval(), evaluation_size)):
        example = dataset.get_item_eval(ind)
        if function(
            model=model, 
            example=example, 
            device=device
        ):
            correct += 1
        total += 1
    
    model.train()
    return correct / total

def evaluate_retiver(
    model: Retriver,
    dataset: Union[RAG_MultiRC_dataset],
    device: torch.device,
    function: Union[
        evaluate_multirc_example_retriver
    ],
    evaluation_size: int = None,
    threshold: Union[float, None] = None,
    threshold_total: float = 0.75,
) -> float:
    '''
    Evaluate a Retriver model on a dataset
    '''
    correct = 0
    total = 0
    if not evaluation_size:
        evaluation_size = dataset.len_eval()

    model.eval()
    for ind in range(min(dataset.len_eval(), evaluation_size)):
        example = dataset.get_item_eval(ind)
        if function(
            model=model,
            example=example,
            device=device,
            threshold=threshold,
            threshold_total=threshold_total
        ):
            correct += 1
        total += 1
    
    model.train()
    return correct / total

RET = TypeVar('RET', bound=Retriver)
def auxilliary_calculate_results_multirc_retriver(
    model: RET,
    example: Tuple[List[Tuple[bool, dict, torch.Tensor]], Tuple[dict, torch.Tensor]],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> List[Tuple[float, bool]]:
    '''
    Calculate the results for a single MultiRC example
    '''
    docs, question = example
    results = []
    with torch.no_grad():
        q_enc = model.forward(question[0], question=True)
    for (cor,doc,_) in docs:
        with torch.no_grad():
            doc_enc = model.forward(doc, question=False)
            res = model.compare(doc_enc, q_enc)
        results.append((res, cor))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results
        

def p_at_k(
    results: List[Tuple[float, bool]],
    device: torch.device,
    k: int = 5
) -> float:
    '''
    Calculate the precision at k
    '''
    number_of_correct = sum([x[1] for x in results])
    return sum([x[1] for x in results[:k]]) / min(k, number_of_correct)

def ap_at_k(
    results: List[Tuple[float, bool]],
    device: torch.device,
    k: int = 5
) -> float:
    '''
    Calculate the average precision at k
    '''
    return sum([p_at_k(results, device, i+1) for i in range(k)]) / k


def calculate_auroc_multirch_retriver_example(
    results: List[Tuple[float, bool]],
    device: torch.device,
):
    '''
    Calculate the AUROC for a single MultiRC example
    '''
    return roc_auc_score([x[1] for x in results], [x[0].cpu() for x in results])

def calculate_aupr_multirc_retriver_example(
    results: List[Tuple[float, bool]],
    device: torch.device,
):
    '''
    Calculate the AUPR for a single MultiRC example
    '''
    y_true = [x[1] for x in results]
    y_score = [x[0].cpu() for x in results]
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def calc_avg_metric_multirc_retriver(
    model: RET,
    dataset: Union[RAG_MultiRC_dataset, RAG_MultiRC_dataset_TL],
    device: torch.device,
    functions: List[callable],
    evaluation_size: int = None
) -> float:
    '''
    Calculate the average AUROC for a dataset
    '''
    res = []
    if not evaluation_size:
        evaluation_size = dataset.len_eval()

    model.eval()
    for ind in range(min(dataset.len_eval(), evaluation_size)):
        example = dataset.get_item_eval(ind)
        results = auxilliary_calculate_results_multirc_retriver(model, example)
        rs = []
        for function in functions:
            rs.append(function(results, device))
        res.append(rs)
    
    model.train()
    res = np.array(res)
    return list(np.mean(res, axis=0))
        