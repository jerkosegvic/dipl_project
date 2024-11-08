import torch
from transformers import AutoTokenizer
from typing import Tuple, Union, Literal, List, Dict
from auxiliary import MultiRC_question, ReCoRD_question

class Boolq_dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        passages: List[str], 
        questions: List[str], 
        answers: List[str]
    ) -> None:
        pass

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns four tensors:
        one with correct answer(True or False) with corresponding attention mask, 
        the other with incorrect(also True) with corresponding attention mask
        The first one is correct
        '''
        pass    

class MultiRC_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        passages: List[str],
        question_answer_objs: List[MultiRC_question]
    ) -> None:
        pass

    def __getitem__(
        self, 
        index:int
    ) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        '''
        Returns if the answer is correct or no,
        tensor for text and tensor for attention mask
        '''
        pass

class RACE_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        passages: List[str],
        passage_inds: List[int],
        questions: List[str],
        answers: List[str],
        correst_answer_ind: List[str]
    ) -> None:
        pass

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[int, 
               torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor
            ]:
        '''
        Returns index of correct answer and eight tensors,
        for every of four possible answers, 
        one tensor for the option, and other for attention mask
        '''
        pass

class ReCoRD_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        passages: List[str],
        question_answer_objs: List[ReCoRD_question]
    ) -> None:
        pass

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        '''
        Returns index of word that needs to be guessed, 
        tensor for text and tensor for attention mask
        '''
        pass