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
        max_length: int = 1024
    ) -> None:
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.encodings = []
        self.corrects = []
        self.max_length = max_length
        for (p,q,a) in zip(passages, questions, answers):
            q_enc = tokenizer(p+q+"?", return_tensors='pt', add_special_tokens=False)

            if q_enc["input_ids"][0].shape[0] > max_length:
                continue

            ans_pos, ans_neg = tokenizer("Yes.", return_tensors='pt', add_special_tokens=False), tokenizer("No.", return_tensors='pt', add_special_tokens=False)
            pos = tokenizer(p+q+"? Yes.", return_tensors='pt', add_special_tokens=False, max_length=max_length, padding="max_length")
            neg = tokenizer(p+q+"? No.", return_tensors='pt', add_special_tokens=False, max_length=max_length, padding="max_length")
            attn_pos = torch.full(pos["input_ids"][0].shape, 0)
            targets_pos = torch.full(pos["input_ids"][0].shape, -100)
            attn_neg = torch.full(neg["input_ids"][0].shape, 0)
            targets_neg = torch.full(neg["input_ids"][0].shape, -100)

            answer_start = q_enc["input_ids"][0].shape[0]
            answer_len_pos = ans_pos["input_ids"][0].shape[0]
            answer_len_neg = ans_neg["input_ids"][0].shape[0]

            targets_pos[answer_start:answer_start+answer_len_pos] = pos["input_ids"][0][answer_start:answer_start+answer_len_pos]
            targets_neg[answer_start:answer_start+answer_len_neg] = neg["input_ids"][0][answer_start:answer_start+answer_len_neg]
            attn_neg[:answer_start+answer_len_neg] = 1
            attn_pos[:answer_start+answer_len_pos] = 1

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
        tokenizer: AutoTokenizer,
        max_length: int = 1024
    ) -> None:
        self.passages = passages
        self.question_answer_objs = question_answer_objs
        self.encodings = []
        self.corrects = []
        self.max_length = max_length
        for (p, q) in zip(passages, question_answer_objs):
            entry = []
            answer_start = tokenizer(''.join(p) + q.question + "?", return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
            for (a, c) in zip(q.answers, q.correct):
                enc = tokenizer(''.join(p) + q.question+"? " + a, return_tensors='pt', add_special_tokens=False, max_length=max_length, padding="max_length")
                
                if enc['input_ids'][0].shape[0] > max_length:
                    continue
                
                attn_mask = torch.full(enc['input_ids'][0].shape, 0)
                targets = torch.full(enc['input_ids'][0].shape, -100)

                answer_len = tokenizer(a, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                attn_mask[:answer_start+answer_len] = 1
                targets[answer_start:answer_start+answer_len] = enc['input_ids'][0][answer_start:answer_start+answer_len]
                entry.append((c, enc['input_ids'], attn_mask, targets))
                if c:
                    self.corrects.append((enc['input_ids'], attn_mask, targets))
            
            if len(entry) == 4:
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
        tokenizer: AutoTokenizer,
        max_length: int = 1024
    ) -> None:
        self.passages = passages
        self.passage_inds = passage_inds
        self.questions = questions
        self.answers = answers
        self.correct_answer_ind = correct_answer_ind
        self.encodings = []
        self.corrects = []
        self.max_length = max_length
        for (p_i, q, a, c) in zip(passage_inds, questions, answers, correct_answer_ind):
            tmp = []
            p = passages[p_i]
            
            q_cpy = q
            if q.count("_") == 1:
                ind = q.index("_")
                answer_start = tokenizer(p + q[:ind], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]

            else:
                answer_start = tokenizer(p + q, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                q_cpy += "_"

            good = True
            for (i, ans) in enumerate(a):
                option = tokenizer(p + q_cpy.replace("_", " " + ans + " "), add_special_tokens=False, return_tensors='pt', max_length=max_length, padding="max_length")
                
                if option['input_ids'][0].shape[0] > max_length:
                    good = False
                    #print("Too long, skipping")
                    continue
                
                answer_len = tokenizer(ans, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                targets = torch.full(option['input_ids'][0].shape, -100)
                attention_mask[:answer_start+answer_len] = 1
                targets[answer_start:answer_start+answer_len] = option['input_ids'][0][answer_start:answer_start+answer_len]
                
                tmp.append((option['input_ids'], attention_mask, targets))
                if i == c:
                    self.corrects.append((option['input_ids'], attention_mask, targets))
            if good:
                self.encodings.append((c, tmp))

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
        tokenizer: AutoTokenizer,
        max_length: int = 1024
    ) -> None:
        self.passages = passages
        self.question_answer_objs = question_answer_objs
        self.encodings = []
        self.corrects = []
        self.max_length = max_length
        for q in question_answer_objs:
            p = passages[q.paragraph_id]
            if p[-1] != ".":
                p += "."
            answer_start = tokenizer(p + q.query[:q.w_ind[0]], return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
            tmp = []
            for (a, span) in zip(q.answers, q.answers_span):
                option = tokenizer(p + " " + q.query[:q.w_ind[0]] + " " + a + " " + q.query[q.w_ind[1]:], return_tensors='pt', 
                                   add_special_tokens=False, max_length=max_length, padding="max_length")
                
                if option['input_ids'][0].shape[0] > max_length:
                    continue
                
                answer_len = tokenizer(a, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                targets = torch.full(option['input_ids'][0].shape, -100)
                attention_mask[:answer_start+answer_len] = 1
                targets[answer_start:answer_start+answer_len] = option['input_ids'][0][answer_start:answer_start+answer_len]
                tmp.append((option['input_ids'], attention_mask, targets))
                self.corrects.append((option['input_ids'], attention_mask, targets))

            if len(tmp) > 0:
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