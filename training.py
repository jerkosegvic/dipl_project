from datasets import Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset, RAG_MultiRC_dataset
from models import Model_LLM, Retriver
from typing import List, Tuple, Union
import torch
import time
from evaluators import evaluate_task_llm, evaluate_retiver

def format_time(elapsed: float) -> str:
    '''
    Format the elapsed time in seconds to DD.MM.YYYY HH:MM:SS format
    '''
    elapsed_rounded = int(round(elapsed))
    return time.strftime("%d.%m.%Y %H:%M:%S", time.gmtime(elapsed_rounded + 3600))

def train_LLM(
        model: Model_LLM,
        dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        eval_dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 8,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        evaluation_interval: int = 100,
        log_interval: int = 100,
        scheduler: torch.optim.lr_scheduler = None,
        eval_func: Union[callable, None] = None,
        train_eval_size: int = None
    ) -> None:
    '''
    Train a Language Model on a dataset
    '''
    model.to(device)
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #breakpoint()
    if eval_func:
        print("Evaluating zero-shot performance...")
        rs_1 = evaluate_task_llm(model, eval_dataset, device, eval_func)
        print(f'    Zero-shot performance eval: {rs_1}')
        if train_eval_size:
            rs_2 = evaluate_task_llm(model, dataset, device, eval_func, train_eval_size)
            print(f'    Zero-shot performance on train: {rs_2}')

    print(f"Training on {len(dataloader)} batches")
    for epoch in range(epochs):
        for (i,batch) in enumerate(dataloader):
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss, logits = model(input_ids, attention_mask, targets)
            loss.backward()
            optimizer.step()

            if i % evaluation_interval == 0:
                print(f'[EVAL] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
                rs = evaluate_LLM(model, eval_dataset, device, batch_size)
                print(f'    Batch: {i} / {len(dataloader)}, Batch Loss: {loss.item()}, Eval loss: {rs}')

            elif i % log_interval == 0:
                print(f'[LOG] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
                print(f'    Batch: {i} / {len(dataloader)}, Batch Loss: {loss.item()}')

        if scheduler:
            scheduler.step()

        if eval_func:
            print(f'[EPOCH EVAL] Timestamp: {format_time(time.time())}) Epoch: {epoch + 1} / {epochs}')
            rs_1 = evaluate_task_llm(model, eval_dataset, device, eval_func)
            rs_2 = evaluate_LLM(model, eval_dataset, device, batch_size)
            print(f'    Eval results: {rs_1}, Eval loss: {rs_2}')
            if train_eval_size:
                rs_3 = evaluate_task_llm(model, dataset, device, eval_func, train_eval_size)
                print(f'    Eval results on train dataset: {rs_3}')

        else:
            print(f'[EPOCH EVAL] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
            rs = evaluate_LLM(model, eval_dataset, device, batch_size)
            print(f'    Eval loss: {rs}')

def train_Retriver(
        model: Retriver,
        dataset: Union[RAG_MultiRC_dataset],
        eval_dataset: Union[RAG_MultiRC_dataset],
        epochs: int,
        train_func: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 8,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        evaluation_interval: int = 100,
        log_interval: int = 100,
        scheduler: torch.optim.lr_scheduler = None,
        eval_func: Union[callable, None] = None,
        train_eval_size: int = None
    ) -> None:
    '''
    Train a Retriver on a dataset using a train_func
    '''
    model.to(device)
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #breakpoint()
    if eval_func:
        print("Evaluating zero-shot performance...")
        rs_1 = evaluate_retriver_loss(model, eval_dataset, train_func, device, batch_size)
        print(f'    Zero-shot performance loss on eval: {rs_1}')        
        rs_2 = evaluate_retiver(model, eval_dataset, device, eval_func)
        print(f'    Zero-shot performance eval: {rs_2}')
        if train_eval_size:
            rs_2 = evaluate_retiver(model, dataset, device, eval_func, evaluation_size=train_eval_size)
            print(f'    Zero-shot performance on train: {rs_2}')
    
    print(f"Training on {len(dataloader)} batches")
    for epoch in range(epochs):
        for (i,batch) in enumerate(dataloader):
            input = batch
            optimizer.zero_grad()
            loss = train_func(model, input, device)
            loss.backward()
            optimizer.step()

            if i % evaluation_interval == 0:
                print(f'[EVAL] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
                rs = evaluate_retriver_loss(model, eval_dataset, train_func, device, batch_size)
                print(f'    Batch: {i} / {len(dataloader)}, Batch Loss: {loss.item()}, Eval loss: {rs}')

            elif i % log_interval == 0:
                print(f'[LOG] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
                print(f'    Batch: {i} / {len(dataloader)}, Batch Loss: {loss.item()}')

        if scheduler:
            scheduler.step()

        if eval_func:
            print(f'[EPOCH EVAL] Timestamp: {format_time(time.time())}) Epoch: {epoch + 1} / {epochs}')
            rs_1 = evaluate_retriver_loss(model, eval_dataset, train_func, device, batch_size)
            rs_2 = evaluate_retiver(model, eval_dataset, device, eval_func)
            print(f'    Eval results: {rs_2}, Eval loss: {rs_1}')
            if train_eval_size:
                rs_3 = evaluate_retiver(model, dataset, device, eval_func, evaluation_size=train_eval_size)
                print(f'    Eval results on train dataset: {rs_3}')

        else:
            print(f'[EPOCH EVAL] Timestamp: {format_time(time.time())}, Epoch: {epoch + 1} / {epochs}')
            rs = evaluate_retriver_loss(model, eval_dataset, train_func, device, batch_size)
            print(f'    Eval loss: {rs}')

            
def evaluate_LLM(
        model: Model_LLM,
        dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        device: torch.device,
        batch_size: int,
    ) -> float:
    '''
    Evaluate a Language Model on a dataset
    '''
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    loss = 0
    print(f"    Evaluating on {len(dataloader)} batches...")
    model.eval()
    for (i,batch) in enumerate(dataloader):
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            loss += model(input_ids, attention_mask, targets)[0].item()

    model.train()
    return loss / len(dataloader)

def evaluate_retriver_loss(
        model: Retriver,
        dataset: Union[RAG_MultiRC_dataset],
        eval_func: callable,
        device: torch.device,
        batch_size: int,
    ) -> float:
    '''
    Evaluate a Retriver model on a dataset
    '''
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"    Evaluating on {len(dataloader)} batches...")
    model.eval()
    loss = 0
    for (i,batch) in enumerate(dataloader):
        input = batch
        with torch.no_grad():
            loss += eval_func(model, input, device).item()

    model.train()
    return loss / len(dataloader)