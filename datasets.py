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
            ln = q_enc["input_ids"][0].shape[0]
            
            if ln > max_length:
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
            attn_neg[:ln] = 1
            attn_pos[:ln] = 1

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
        self.passages = []
        self.question_answer_objs = []
        self.encodings = []
        self.corrects = []
        self.max_length = max_length
        for (p, q) in zip(passages, question_answer_objs):
            entry = []
            bs = ''.join(p) + q.question
            if bs[-1] != "?":
                bs += "?"
            answer_start = tokenizer(bs, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]

            save = False
            for (a, c) in zip(q.answers, q.correct):
                bs_a = bs + a
                enc = tokenizer(bs_a, return_tensors='pt', add_special_tokens=False, max_length=max_length, padding="max_length")
                answer_len = tokenizer(a, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                ln = answer_len + answer_start

                if ln > max_length:
                    continue
                
                attn_mask = torch.full(enc['input_ids'][0].shape, 0)
                targets = torch.full(enc['input_ids'][0].shape, -100)
                
                attn_mask[:ln] = 1
                targets[answer_start:answer_start+answer_len] = enc['input_ids'][0][answer_start:answer_start+answer_len]
                entry.append((c, enc['input_ids'], attn_mask, targets))
                
                if c:
                    self.corrects.append((enc['input_ids'], attn_mask, targets))
                    save = True

            # save variable defines if something is added to training list (corrects), if not, the example is skipped
            if save:
                self.encodings.append(entry)
                self.passages.append(p)
                self.question_answer_objs.append(q)

    def class_name() -> str:
        return "multirc_base"
    
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
                answer_len = tokenizer(ans, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                ln = tokenizer(p + q_cpy.replace("_", " " + ans + " "), add_special_tokens=False, return_tensors='pt')['input_ids'][0].shape[0]

                if ln > max_length:
                    good = False
                    #print("Too long, skipping")
                    continue
                
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                targets = torch.full(option['input_ids'][0].shape, -100)
                attention_mask[:ln] = 1
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
                ln = option['input_ids'][0].shape[0]
                if ln > max_length:
                    continue
                
                answer_len = tokenizer(a, return_tensors='pt', add_special_tokens=False)['input_ids'][0].shape[0]
                attention_mask = torch.full(option['input_ids'][0].shape, 0)
                targets = torch.full(option['input_ids'][0].shape, -100)
                attention_mask[:ln] = 1
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
    
class RAG_MultiRC_dataset(MultiRC_dataset):
    def __init__(
        self,
        passages: List[List[str]],
        question_answer_objs: List[MultiRC_question],
        tokenizer_llm: AutoTokenizer,
        tokenizer_retrive: AutoTokenizer,
        max_length: int = 1024,
        max_length_enc: int = 512
    ) -> None:
        super(RAG_MultiRC_dataset, self).__init__(
            passages=passages, 
            question_answer_objs=question_answer_objs, 
            tokenizer=tokenizer_llm,
            max_length=max_length
        )
        self.questions_enc = []
        self.docs_enc = []
        self.docs_enc_all = []
        self.max_length_enc = max_length_enc
        for (j,q) in enumerate(self.question_answer_objs):
            sentences = self.passages[q.passage_index]
            q_enc_ret = tokenizer_retrive(q.question, return_tensors='pt', add_special_tokens=True, padding="max_length", max_length=self.max_length_enc)
            q_enc_gen = tokenizer_llm(q.question, return_tensors='pt', add_special_tokens=False)['input_ids']
            self.questions_enc.append((q_enc_ret, q_enc_gen))
            obj = []
            for (i,p) in enumerate(sentences):
                p_enc_ret = tokenizer_retrive(p, return_tensors='pt', add_special_tokens=True, padding="max_length", max_length=self.max_length_enc)
                p_enc_gen = tokenizer_llm(p, return_tensors='pt', add_special_tokens=False)['input_ids']
                cor = True if i in q.sentences_used else False
                self.docs_enc_all.append((cor, j, p_enc_ret, p_enc_gen))
                obj.append((cor, p_enc_ret, p_enc_gen))

            self.docs_enc.append(obj)

    def class_name() -> str:
        return "rag_multirc_base"

    def __getitem__(
        self, 
        index: int
    ) -> Tuple[bool, dict, dict]:
        '''
        Returns boolean value(True if sentence is relevant), 
        encoding dictionary for sentence and encoding dictionary for question
        '''
        return self.docs_enc_all[index][0], self.docs_enc_all[index][2], self.questions_enc[self.docs_enc_all[index][1]][0]

    def __len__(
        self
    ) -> int:
        return len(self.docs_enc_all)

    def get_item_eval(
        self,
        index: int
    ) -> Tuple[List[Tuple[bool, dict, torch.Tensor]], Tuple[dict, torch.Tensor]]:
        '''
        Returns Tuple of two elements.
        First is a list of tuples (one for each sentence), 
        where the first element is boolean value(True if sentence is relevant),
        the second element is encoding dictionary for sentence and the third element is encoding tensor
        of input ids for text generation.
        The second element is encoding dictionary for question and encoding tensor of input ids for text generation.
        '''
        return (self.docs_enc[index], self.questions_enc[index])

    def len_eval(
        self
    ) -> int:
        return len(self.docs_enc)

class RAG_MultiRC_dataset_TL(RAG_MultiRC_dataset):
    def __init__(
        self,
        passages: List[List[str]],
        question_answer_objs: List[MultiRC_question],
        tokenizer_llm: AutoTokenizer,
        tokenizer_retrive: AutoTokenizer,
        max_length: int = 1024,
        max_length_enc: int = 512
    ) -> None:
        super(RAG_MultiRC_dataset_TL, self).__init__(
            passages=passages, 
            question_answer_objs=question_answer_objs, 
            tokenizer_llm=tokenizer_llm,
            tokenizer_retrive=tokenizer_retrive,
            max_length=max_length,
            max_length_enc=max_length_enc
        )
        self.triplets = []
        for (q, obj) in zip(self.questions_enc, self.docs_enc):
            positives = list(filter(lambda x: x[0] == True, obj))
            negatives = list(filter(lambda x: x[0] == False, obj))
            for p in positives:
                for n in negatives:
                    self.triplets.append((q[0], p[1], n[1]))
    
    def class_name() -> str:
        return "rag_multirc_triplet_loss"
    
    def __getitem__(
        self, 
        index: int
    ) -> Tuple[dict, dict, dict]:
        '''
        Returns three encoding dictionaries, one for anchor, one for positive and one for negative
        '''
        return self.triplets[index]
    
    def __len__(
        self
    ) -> int:
        return len(self.triplets)