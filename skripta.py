from datasets import Boolq_dataset
from dataset_loaders import load_boolq
from transformers import BertTokenizer

if __name__ == '__main__':
    dataset = load_boolq('data/boolq/train.jsonl', Dataset_=Boolq_dataset, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))