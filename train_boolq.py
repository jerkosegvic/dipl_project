from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from models import Model_LLM
from training import train_LLM, evaluate_LLM
from datasets import Boolq_dataset
from dataset_loaders import load_boolq
from evaluators import evaluate_boolqe_example_llm
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-4
MODEL_NAME = 'gpt2'
BATCH_SIZE = 4
EPOCHS = 5

if __name__ == '__main__':
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = Model_LLM(GPT2LMHeadModel.from_pretrained(MODEL_NAME), tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    print("Model initialized")
    train_dataset = load_boolq(
        'raw_data/BoolQ/train.jsonl',
        tokenizer,
        Dataset_=Boolq_dataset,
        max_length=1024
    )
    print(f"Train dataset loaded, with length {len(train_dataset)}")
    val_dataset = load_boolq(
        'raw_data/BoolQ/dev.jsonl',
        tokenizer,
        Dataset_=Boolq_dataset,
        max_length=1024
    )
    print(f"Validation dataset loaded, with length {len(val_dataset)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Optimizer initialized")
    print("Starting training")
    train_LLM(
        model,
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        evaluation_interval=1000,
        log_interval=200,
        eval_func=evaluate_boolqe_example_llm
    )
    print("Training finished")
    print("Saving model")
    torch.save(model.state_dict(), 'models/boolq_model.pth')