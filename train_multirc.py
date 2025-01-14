from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from models import Model_LLM
from training import train_LLM, evaluate_LLM
from datasets import MultiRC_dataset
from dataset_loaders import load_multirc
from evaluators import evaluate_multirc_example_llm
import torch
from functools import partial

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-5
MODEL_NAME = 'gpt2'
BATCH_SIZE = 3
EPOCHS = 10
MAIN_THRESHOLD = 0.85
SCHEDULER = True
GAMMA = 0.7

if __name__ == '__main__':
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = Model_LLM(GPT2LMHeadModel.from_pretrained(MODEL_NAME), tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id 
    print("Model initialized")
    train_dataset = load_multirc(
        'raw_data/multirc-v2/splitv2/train_456-fixedIds.json',
        tokenizer,
        Dataset_=MultiRC_dataset,
        max_length=1024
    )
    print(f"Train dataset loaded, with length {len(train_dataset)}")
    val_dataset = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer,
        Dataset_=MultiRC_dataset,
        max_length=1024
    )
    print(f"Validation dataset loaded, with length {len(val_dataset)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
        print("Scheduler initialized")
    else:
        scheduler = None

    print("Optimizer initialized")
    print("Starting training")
    func_ = partial(evaluate_multirc_example_llm, threshold_total=MAIN_THRESHOLD)
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
        eval_func=func_,
        train_eval_size=1000,
        scheduler=scheduler
    )
    print("Training finished")
    print("Saving model")
    model.eval()
    model.to('cpu')
    torch.save(model, 'models/multirc_model.pth')
