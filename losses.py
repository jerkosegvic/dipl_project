import torch
from models import Retriver
from typing import Tuple

def train_func_retriver_multiRC(
        model: Retriver, 
        input: Tuple[bool, dict, dict], 
        device: torch.device
    ) -> torch.Tensor:
    '''
    Train function for MultiRC dataset
    '''
    is_correct, question, answer = input
    question = {k: v.to(device) for k,v in question.items()}
    answer = {k: v.to(device) for k,v in answer.items()}
    
    q_enc = model.forward(question)
    a_enc = model.forward(answer)
    
    loss = torch.nn.functional.cosine_embedding_loss(q_enc, a_enc, torch.tensor([1.0 if i else -1.0 for i in is_correct]).to(device))
    #print(f'    Loss: {loss}, target: {is_correct}')
    return loss.mean()

def triplet_loss_func_multiRC(
        model: Retriver, 
        input: Tuple[dict, dict, dict], 
        device: torch.device,
        margin: float = 1.0,
        p_norm: int = 2
    ) -> torch.Tensor:
    '''
    Train function for MultiRC dataset
    '''
    anchor, positive, negative = input
    anchor = {k: v.to(device) for k,v in anchor.items()}
    positive = {k: v.to(device) for k,v in positive.items()}
    negative = {k: v.to(device) for k,v in negative.items()}

    a_enc = model.forward(anchor)
    p_enc = model.forward(positive) 
    n_enc = model.forward(negative)

    loss = torch.nn.functional.triplet_margin_loss(a_enc, p_enc, n_enc, margin=margin, p=p_norm)
    return loss.mean()