import torch
from transformers import AutoTokenizer
from typing import Tuple, Union, Literal, List, Dict
from auxiliary import MultiRC_question, ReCoRD_question

class Boolq_dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        passages: List[str], 
        questions: List[str], 
        answers: List[bool],
        tokenizer: AutoTokenizer,
    ) -> None:
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.encodings = []
        for (p,q,a) in zip(passages, questions, answers):
            ans_pos, ans_neg = tokenizer("Yes.", return_tensors='pt'), tokenizer("No.", return_tensors='pt')
            pos, neg = tokenizer(p+q+"? Yes.", return_tensors='pt'), tokenizer(p+q+"? No.", return_tensors='pt')
            attn_pos = torch.full(pos["input_ids"][0].shape, 0)
            targets_pos = torch.full(pos["input_ids"][0].shape, -100)
            attn_neg = torch.full(neg["input_ids"][0].shape, 0)
            targets_neg = torch.full(neg["input_ids"][0].shape, -100)
            answer_len_pos = ans_pos["input_ids"][0].shape[0] - 1
            answer_len_neg = ans_neg["input_ids"][0].shape[0] - 1
            targets_pos[-answer_len_pos:] = pos["input_ids"][0][-answer_len_pos:]
            targets_neg[-answer_len_neg:] = neg["input_ids"][0][-answer_len_neg:]
            attn_neg[-answer_len_neg:] = 1
            attn_pos[-answer_len_pos:] = 1
            if a == True:
                self.encodings.append((pos['input_ids'], attn_pos, targets_pos, neg['input_ids'], attn_neg, targets_neg))
            else:
                self.encodings.append((neg['input_ids'], attn_neg, targets_neg, pos['input_ids'], attn_pos, targets_pos))

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns six tensors:
        one with correct answer(True or False) with corresponding attention mask and targets, 
        the other with incorrect(also True) with corresponding attention mask and targets
        The first one is correct
        '''
        return self.encodings[index]

class MultiRC_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        passages: List[List[str]],
        question_answer_objs: List[MultiRC_question],
        tokenizer: AutoTokenizer
    ) -> None:
        self.passages = passages
        self.question_answer_objs = question_answer_objs
        self.encodings = []
        self.corrects = []
        for (p, q) in zip(passages, question_answer_objs):
            entry = []
            for (a, c) in zip(q.answers, q.correct):
                enc = tokenizer(''.join(p) + q.question+"? " + a, return_tensors='pt')
                ans_len = tokenizer(a, return_tensors='pt')['input_ids'][0].shape[0]
                attn_mask = torch.full(enc['input_ids'][0].shape, 0)
                targets = torch.full(enc['input_ids'][0].shape, -100)
                attn_mask[-ans_len:] = 1
                targets[-ans_len:] = enc['input_ids'][0][-ans_len:]
                entry.append((c, enc['input_ids'], attn_mask, targets))
                if c:
                    self.corrects.append((enc['input_ids'], attn_mask, targets))

            self.encodings.append(entry)

    def __getitem__(
        self, 
        index:int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns one tensor for text, tensor for attention mask and tensor for targets.
        This method returns correct answers only.
        For evaluation purposes, the correct answers are stored in a separate list
        '''
        return self.corrects[index]

    def get_item_eval(
        self,
        index: int
    ) -> List[Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Returns list of tuples, where the first element is boolean value(True if answer is correct),
        and the other three are tensors for texts, attention mask and targets
        '''
        return self.encodings[index]

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