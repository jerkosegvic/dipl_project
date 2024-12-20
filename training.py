from datasets import Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset
from models import Model_LLM
from typing import List, Tuple, Union
import torch

def train_LLM(
        model: Model_LLM,
        dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        eval_dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        epochs: int,
        optimizer: torch.optim.Optimizer,
        batch_size: int = 8,
        lr: float = 1e-5,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        evaluation_interval: int = 100,
        scheduler: torch.optim.lr_scheduler = None,
        scheduler_step: int = None,
        scheduler_gamma: float = None,
        eval_func: Union[callable, None] = None
    ) -> None:
    '''
    Train a Language Model on a dataset
    '''
    model.to(device)
    model.train()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
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
                rs = evaluate_LLM(model, eval_dataset, device, batch_size)
                print(f'Epoch: {epoch}, Batch: {i} / {len(dataloader)}, Loss: {loss.item()}, Eval: {rs}')

        if scheduler and scheduler_step and scheduler_gamma:
            if epoch % scheduler_step == 0:
                scheduler.step()

        if eval_func:
            rs_1 = eval_func(model, eval_dataset)    
            rs_2 = evaluate_LLM(model, eval_dataset, device, batch_size)
            print(f'Epoch: {epoch} / {epochs}, Eval results: {rs_1}, Eval loss: {rs_2}')

        else:
            rs = evaluate_LLM(model, eval_dataset, device, batch_size)
            print(f'Epoch: {epoch} / {epochs}, Eval: {rs}')

def evaluate_LLM(
        model: Model_LLM,
        dataset: Union[Boolq_dataset, MultiRC_dataset, RACE_dataset, ReCoRD_dataset],
        device: torch.device,
        batch_size: int
    ) -> float:
    '''
    Evaluate a Language Model on a dataset
    '''
    model.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss = 0
    for (i,batch) in enumerate(dataloader):
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        loss += model(input_ids, attention_mask, targets)[0]

    return loss / len(dataloader)