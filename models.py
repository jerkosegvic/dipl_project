import torch
from transformers import BertModel, GPT2Model, AutoTokenizer
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
