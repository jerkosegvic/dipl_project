from transformers import AutoTokenizer, BertModel, BertTokenizer, GPT2Tokenizer
from models import Retriver, Retriver_TL
from training import train_Retriver
from datasets import RAG_MultiRC_dataset, RAG_MultiRC_dataset_TL
from dataset_loaders import load_multirc
from evaluators import evaluate_multirc_example_retriver
import torch
from functools import partial
from losses import train_func_retriver_multiRC, triplet_loss_func_multiRC


# CONFIG FOR BASIC CONTRASTIVE LEARNING TRAINING
"""
TRAIN_FUNC = train_func_retriver_multiRC
DATASET = RAG_MultiRC_dataset
RETRIVER_MODEL = Retriver
SAVE_NAME = 'retriver_multirc'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-5
MODEL_NAME = 'gpt2'
RETRIVER_NAME = 'bert-base-uncased'
BATCH_SIZE = 4
EPOCHS = 5
MAIN_THRESHOLD = 0.85
SCHEDULER = True
GAMMA = 0.7
"""

# CONFIG FOR TRIPLET LOSS TRAINING

TRAIN_FUNC = triplet_loss_func_multiRC
DATASET = RAG_MultiRC_dataset_TL
RETRIVER_MODEL = Retriver_TL
SAVE_NAME = 'retriver_multirc_TL'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-5
MODEL_NAME = 'gpt2'
RETRIVER_NAME = 'bert-base-uncased'
BATCH_SIZE = 4
EPOCHS = 3
MAIN_THRESHOLD = 0.85
SCHEDULER = True
GAMMA = 0.7


if __name__ == '__main__':
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    tokenizer_llm = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
        tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id
    tokenizer = BertTokenizer.from_pretrained(RETRIVER_NAME)
    model = Retriver(BertModel.from_pretrained(RETRIVER_NAME), tokenizer)
    print("Model initialized")
    train_dataset = load_multirc(
        'raw_data/multirc-v2/splitv2/train_456-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=DATASET,
        max_length=1024,
        max_length_rag=512
    )
    print(f"Train dataset loaded, with length {len(train_dataset)}")
    val_dataset = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=DATASET,
        max_length=1024,
        max_length_rag=512
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
    func_ = partial(evaluate_multirc_example_retriver, threshold_total=MAIN_THRESHOLD)
    train_Retriver(
        model,
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        train_func=TRAIN_FUNC,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        scheduler=scheduler,
        eval_func=func_,
        train_eval_size=500,
        evaluation_interval=1000,
        log_interval=200
    )
    print("Training complete")
    torch.save(model, f'models/{SAVE_NAME}.pth')
    print("Model saved")