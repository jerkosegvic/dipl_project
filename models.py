import torch
from transformers import BertModel, GPT2Model, AutoTokenizer, RobertaModel
from typing import List, Tuple, Union, Type, TypeVar

class Model_LLM(torch.nn.Module):
    def __init__(
        self,
        model: Union[BertModel, GPT2Model],
        tokenizer: AutoTokenizer
    ) -> None:
        super(Model_LLM, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        return outputs.loss, outputs.logits
    
    def decode(
        self,
        input_ids: torch.Tensor,
        top_k: int = 5
    ) -> str:
        ##TODO: implement decoding
        pass

class Retriver(torch.nn.Module):
    def __init__(
        self,
        model: Union[BertModel, RobertaModel],
        tokenizer: AutoTokenizer
    ) -> None:
        super(Retriver, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        input_ids = input['input_ids'].to(device)
        attention_mask = input['attention_mask'].to(device)
        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:,0,:]
    
    def compare(
        self,
        doc: torch.Tensor,
        question: torch.Tensor
    ) -> torch.Tensor:
        if doc.shape.__len__() == 1:
            doc = doc.unsqueeze(0)
        if question.shape.__len__() == 1:
            question = question.unsqueeze(0)
        return torch.nn.functional.cosine_similarity(doc, question)
    
class Retriver_TL(Retriver):
    def __init__(
        self,
        model: Union[BertModel, RobertaModel],
        tokenizer: AutoTokenizer,
        p_norm: int = 2
    ) -> None:
        super(Retriver_TL, self).__init__(model, tokenizer)
        self.p_norm = p_norm

    def compare(
        self,
        doc: torch.Tensor,
        question: torch.Tensor
    ) -> torch.Tensor:
        if doc.shape.__len__() == 1:
            doc = doc.unsqueeze(0)
        if question.shape.__len__() == 1:
            question = question.unsqueeze(0)
        
        loss = torch.nn.functional.pairwise_distance(doc, question, p=self.p_norm)