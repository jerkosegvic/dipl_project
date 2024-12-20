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
        self.corrects = []
        for (p,q,a) in zip(passages, questions, answers):
            ans_pos, ans_neg = tokenizer("Yes.", return_tensors='pt', add_special_tokens=False), tokenizer("No.", return_tensors='pt', add_special_tokens=False)
            pos, neg = tokenizer(p+q+"? Yes.", return_tensors='pt', add_special_tokens=False), tokenizer(p+q+"? No.", return_tensors='pt', add_special_tokens=False)
            attn_pos = torch.full(pos["input_ids"][0].shape, 0)
            targets_pos = torch.full(pos["input_ids"][0].shape, -100)
            attn_neg = torch.full(neg["input_ids"][0].shape, 0)
            targets_neg = torch.full(neg["input_ids"][0].shape, -100)
            answer_len_pos = ans_pos["input_ids"][0].shape[0]
            answer_len_neg = ans_neg["input_ids"][0].shape[0]
            targets_pos[-answer_len_pos:] = pos["input_ids"][0][-answer_len_pos:]
            targets_neg[-answer_len_neg:] = neg["input_ids"][0][-answer_len_neg:]
            attn_neg[-answer_len_neg:] = 1
            attn_pos[-answer_len_pos:] = 1
            if a == True:
                self.encodings.append((pos['input_ids'], attn_pos, targets_pos, neg['input_ids'], attn_neg, targets_neg))
                self.corrects.append((pos['input_ids'], attn_pos, targets_pos))
            else:
                self.encodings.append((neg['input_ids'], attn_neg, targets_neg, pos['input_ids'], attn_pos, targets_pos))
                self.corrects.append((neg['input_ids'], attn_neg, targets_neg))
    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns six tensors:
        one with correct answer(True or False) with corresponding attention mask and targets, 
        the other with incorrect(also True) with corresponding attention mask and targets
        The first one is correct
        '''
        return self.corrects[index]
    
    def __len__(
        self
    ) -> int:
        return len(self.corrects)
    
    def len_eval(
        self
    ) -> int:
        return len(self.encodings)
    
    def get_item_eval(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                enc = tokenizer(''.join(p) + q.question+"? " + a, return_tensors='pt', add_special_tokens=False)
                ans_len = tokenizer(a, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
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
    
    def __len__(
        self
    ) -> int:
        return len(self.corrects)
    
    def len_eval(
        self
    ) -> int:
        return len(self.encodings)
    
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
        correct_answer_ind: List[str],
        tokenizer: AutoTokenizer
    ) -> None:
        self.passages = passages
        self.passage_inds = passage_inds
        self.questions = questions
        self.answers = answers
        self.correct_answer_ind = correct_answer_ind
        self.encodings = []
        self.corrects = []
        for (p_i, q, a, c) in zip(passage_inds, questions, answers, correct_answer_ind):
            encodings = []
            p = passages[p_i]
            enc = tokenizer(p + q, return_tensors='pt', add_special_tokens=False)
            
            q_cpy = q
            if q.count("_") == 1:
                ind = q.index("_")
                left_sz = tokenizer(p + q[:ind], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                right_sz = tokenizer(q[ind+1:], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]

            else:
                left_sz = tokenizer(p + q, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                right_sz = 0
                q_cpy += "_"


            for (i, ans) in enumerate(a):
                tmp = []
                option = tokenizer(p + q_cpy.replace("_", " " + ans + " "), add_special_tokens=False, return_tensors='pt')
                sz = option['input_ids'][0].shape[0]
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                attention_mask[left_sz:sz-right_sz] = 1
                targets = torch.full(option['input_ids'][0].shape, -100)
                targets[left_sz:sz-right_sz] = option['input_ids'][0][left_sz:sz-right_sz]
                
                tmp.append((option['input_ids'], attention_mask, targets))
                if i == c:
                    self.corrects.append((option['input_ids'], attention_mask, targets))

            encodings.append((c, tmp))

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        '''
        Returns index of correct answer and eight tensors,
        for every of four possible answers, 
        one tensor for the option, and other for attention mask
        '''
        return self.corrects[index]

    def __len__(
        self
    ) -> int:
        return len(self.corrects)
    
    def len_eval(
        self
    ) -> int:
        return len(self.encodings)
    
    def get_item_eval(
        self,
        index: int
    ) -> Tuple[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        '''
        Returns index of correct answer and list of tuples, where each element is a tuple
        with three tensors, one for option, one for attention mask and one for targets
        '''
        return self.encodings[index]

class ReCoRD_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        passages: List[str],
        question_answer_objs: List[ReCoRD_question],
        tokenizer: AutoTokenizer
    ) -> None:
        self.passages = passages
        self.question_answer_objs = question_answer_objs
        self.encodings = []
        self.corrects = []
        for q in question_answer_objs:
            p = passages[q.paragraph_id]
            if p[-1] != ".":
                p += "."
            left_sz = tokenizer(p + q.query[:q.w_ind[0]], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
            right_sz = tokenizer(q.query[q.w_ind[1]:], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
            tmp = []
            for (a, span) in zip(q.answers, q.answers_span):
                option = tokenizer(p + " " + q.query[:q.w_ind[0]] + " " + a + " " + q.query[q.w_ind[1]:], return_tensors='pt', add_special_tokens=False)
                sz = option['input_ids'][0].shape[0]
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                attention_mask[left_sz:sz-right_sz] = 1
                targets = torch.full(option['input_ids'][0].shape, -100)
                targets[left_sz:sz-right_sz] = option['input_ids'][0][left_sz:sz-right_sz]
                tmp.append((option['input_ids'], attention_mask, targets))
                self.corrects.append((option['input_ids'], attention_mask, targets))

            self.encodings.append(tmp)

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Returns three tensors, one for text, one for attention mask and one for targets
        '''
        return self.corrects[index]
    
    def __len__(
        self
    ) -> int:
        return len(self.corrects)
    
    def len_eval(
        self
    ) -> int:
        return len(self.encodings)
    
    def get_item_eval(
        self,
        index: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        Returns list of tuples, where each element is a tuple with three tensors,
        one for text, one for attention mask and one for targets. The list contains all possible answers
        '''
        return self.encodings[index]