from functools import partial
from models import Retriver, Dual_Retriver, Retriver_TL, Dual_Retriver_TL
from datasets import RAG_MultiRC_dataset, RAG_MultiRC_dataset_TL
from dataset_loaders import load_multirc
from evaluators import evaluate_multirc_example_retriver, ap_at_k, p_at_k, auxilliary_calculate
from transformers import AutoTokenizer, BertModel, BertTokenizer, GPT2Tokenizer, AutoModel
import os
import warnings
import torch
import json
from evaluators import evaluate_multirc_example_retriver, ap_at_k, p_at_k, \
    auxilliary_calculate, calculate_auroc_multirch_retriver_example, calc_avg_metric_multirc_retriver, \
    evaluate_retiver, calculate_aupr_multirc_retriver_example

MODELS_PATH = {
    "retriver_multirc_base": "models/multirc_retrivers/retriver_multirc_base/",
    "retriver_multirc_TL": "models/multirc_retrivers/retriver_multirc_TL/",
    "retriver_multirc_PO": "models/multirc_retrivers/retriver_multirc_PO/",
    "dual_retriver_multirc_base": "models/multirc_retrivers/dual_retriver_multirc_base/",
    "dual_retriver_multirc_TL": "models/multirc_retrivers/dual_retriver_multirc_TL/",
    "dual_retriver_multirc_PO": "models/multirc_retrivers/dual_retriver_multirc_PO/"
}

MODELS_TYPES = {
    "retriver_multirc_base": Retriver,
    "retriver_multirc_TL": Retriver_TL,
    "retriver_multirc_PO": Retriver,
    "dual_retriver_multirc_base": Dual_Retriver,
    "dual_retriver_multirc_TL": Dual_Retriver_TL,
    "dual_retriver_multirc_PO": Dual_Retriver
}

VALID_RESULTS = {}
TEST_RESULTS = {}
RETRIVER_NAME = 'bert-base-uncased'
MODEL_NAME = 'gpt2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(
    model,
    dataset,
):
    rv = {}
    func_ap_at_3 = partial(ap_at_k, k=3)
    func_p_at_1 = partial(p_at_k, k=1)
    func_p_at_3 = partial(p_at_k, k=3)
    avg_auroc, avg_aupr, avg_ap_3, avg_p_1, avg_p_3 = calc_avg_metric_multirc_retriver(
        model=model,
        dataset=dataset,
        device=DEVICE,
        functions=[
            calculate_auroc_multirch_retriver_example,
            calculate_aupr_multirc_retriver_example,
            func_ap_at_3,
            func_p_at_1,
            func_p_at_3,
        ]
    )
    
    print(f"    Average AUROC: {avg_auroc}")
    print(f"    Average AUPR: {avg_aupr}")
    print(f"    MAP@3: {avg_ap_3}")
    print(f"    MP@1: {avg_p_1}")    
    print(f"    MP@3: {avg_p_3}")
    rv["AVG_AUROC"] = avg_auroc
    rv["AVG_AUPR"] = avg_aupr
    rv["MAP@3"] = avg_ap_3
    rv["MP@1"] = avg_p_1
    rv["MP@3"] = avg_p_3

    func_ = partial(evaluate_multirc_example_retriver, threshold_total=0.95)
    result = evaluate_retiver(
        model=model,
        dataset=dataset,
        device=DEVICE,
        function=func_
    )
    print(f"    Eval results @ 0.95: {result}")
    rv["Eval@0.95"] = result

    return rv

def evaluate_versions(
    model_name,
    dataset
):
    VALID_RESULTS[model_name] = {}
    if model_name.count("dual") > 0:
        model = MODELS_TYPES[model_name](
            AutoModel.from_pretrained(RETRIVER_NAME),
            AutoModel.from_pretrained(RETRIVER_NAME),
            AutoTokenizer.from_pretrained(RETRIVER_NAME),
            AutoTokenizer.from_pretrained(RETRIVER_NAME)
        )
        
    else:
        model = MODELS_TYPES[model_name](
            AutoModel.from_pretrained(RETRIVER_NAME),
            AutoTokenizer.from_pretrained(RETRIVER_NAME)
        )

    model_paths = os.listdir(MODELS_PATH[model_name])
    model_paths.sort()
    model_paths_ = [os.path.join(MODELS_PATH[model_name], model_path) for model_path in model_paths]
    
    for (model_path,version) in zip(model_paths_, model_paths):
        model = torch.load(model_path)
        model.eval()
        model.to(DEVICE)
        print(f"Evaluating model {model_path}")
        VALID_RESULTS[model_name][version] = evaluate_model(model, dataset)
        print("")

def create_leaderboard():
    metrics = ["AVG_AUROC", "AVG_AUPR", "MAP@3", "MP@1", "MP@3", "Eval@0.95"]
    overall_ranking = {}

    for model in VALID_RESULTS:
        print(f"Model: {model}")
        model_ranking = {}
        for metric in metrics:
            print(f"  Metric: {metric}")
            sorted_versions = sorted(VALID_RESULTS[model].items(), key=lambda x: x[1][metric], reverse=True)
            for rank, (version, results) in enumerate(sorted_versions, start=1):
                print(f"    {rank} - Version: {version}, {metric}: {results[metric]}")
                if version not in model_ranking:
                    model_ranking[version] = 0
                model_ranking[version] += rank
            print("")
        
        # Print overall ranking for the model
        print(f"  Overall Ranking for Model: {model}")
        sorted_overall = sorted(model_ranking.items(), key=lambda x: x[1])
        for rank, (version, total_rank) in enumerate(sorted_overall, start=1):
            print(f"  {rank} - Version: {version}, Total Rank: {total_rank}")
            if model not in overall_ranking:
                overall_ranking[model] = {}
            overall_ranking[model][version] = total_rank
        print("")

    # Cumulative leaderboard
    cumulative_results = {}
    for metric in metrics:
        cumulative_results[metric] = []
        for model in VALID_RESULTS:
            for version, results in VALID_RESULTS[model].items():
                cumulative_results[metric].append((f"{model}_{version}", results[metric]))

    cumulative_ranking = {}
    for metric in metrics:
        print(f"Cumulative Metric: {metric}")
        sorted_cumulative = sorted(cumulative_results[metric], key=lambda x: x[1], reverse=True)
        for rank, (model_version, result) in enumerate(sorted_cumulative, start=1):
            print(f"  {rank} - Model_Version: {model_version}, {metric}: {result}")
            if model_version not in cumulative_ranking:
                cumulative_ranking[model_version] = 0
            cumulative_ranking[model_version] += rank
        print("")

    # Print overall cumulative ranking
    print("Overall Cumulative Ranking")
    sorted_cumulative_overall = sorted(cumulative_ranking.items(), key=lambda x: x[1])
    for rank, (model_version, total_rank) in enumerate(sorted_cumulative_overall, start=1):
        print(f"  {rank} - Model_Version: {model_version}, Total Rank: {total_rank}")
    print("")

    # Return best model of each category
    best_models = {}
    for model in overall_ranking:
        best_version = min(overall_ranking[model], key=overall_ranking[model].get)
        best_models[model] = best_version
    print("Best Models of Each Category")
    for model, best_version in best_models.items():
        print(f"  Model: {model}, Best Version: {best_version}, Total Rank: {cumulative_ranking[f'{model}_{best_version}']}")
    print("")

    return best_models

def create_leaderboard_test():
    metrics = ["AVG_AUROC", "AVG_AUPR", "MAP@3", "MP@1", "MP@3", "Eval@0.95"]
    cumulative_results = {}
    
    for metric in metrics:
        cumulative_results[metric] = []
        for model, results in TEST_RESULTS.items():
            cumulative_results[metric].append((model, results[metric]))

    cumulative_ranking = {}
    for metric in metrics:
        print(f"Cumulative Metric: {metric}")
        sorted_cumulative = sorted(cumulative_results[metric], key=lambda x: x[1], reverse=True)
        for rank, (model_version, result) in enumerate(sorted_cumulative, start=1):
            print(f"  {rank} - Model_Version: {model_version}, {metric}: {result}")
            if model_version not in cumulative_ranking:
                cumulative_ranking[model_version] = 0
            cumulative_ranking[model_version] += rank
        print("")

    # Print overall cumulative ranking
    print("Overall Cumulative Ranking")
    sorted_cumulative_overall = sorted(cumulative_ranking.items(), key=lambda x: x[1])
    for rank, (model_version, total_rank) in enumerate(sorted_cumulative_overall, start=1):
        print(f"  {rank} - Model_Version: {model_version}, Total Rank: {total_rank}")
    print("")
    
    

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(RETRIVER_NAME)
    tokenizer_llm = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer_llm.pad_token is None:
        tokenizer_llm.pad_token = tokenizer_llm.eos_token
        tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id
    
    validation_dataset_base = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=RAG_MultiRC_dataset,
        max_length=1024,
        max_length_rag=512,
        ind_range=(0, 40)
    )
    validation_dataset_TL = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=RAG_MultiRC_dataset_TL,
        max_length=1024,
        max_length_rag=512,
        ind_range=(0, 40)
    )
    test_dataset_base = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=RAG_MultiRC_dataset,
        max_length=1024,
        max_length_rag=512,
        ind_range=(40, 83)
    )
    test_dataset_TL = load_multirc(
        'raw_data/multirc-v2/splitv2/dev_83-fixedIds.json',
        tokenizer=tokenizer_llm,
        tokenizer_rag=tokenizer,
        Dataset_=RAG_MultiRC_dataset_TL,
        max_length=1024,
        max_length_rag=512,
        ind_range=(40, 83)
    )
    warnings.filterwarnings("ignore")
    
    evaluate_versions("retriver_multirc_base", validation_dataset_base)
    evaluate_versions("retriver_multirc_TL", validation_dataset_TL)
    evaluate_versions("retriver_multirc_PO", validation_dataset_base)
    evaluate_versions("dual_retriver_multirc_base", validation_dataset_base)
    evaluate_versions("dual_retriver_multirc_TL", validation_dataset_TL)
    evaluate_versions("dual_retriver_multirc_PO", validation_dataset_base)
    
    with open('output/valid_results.json', 'w') as f:
        json.dump(VALID_RESULTS, f, indent=4)
    
    best_models = create_leaderboard()

    # Evaluate best models on test set
    for bm in best_models:
        model_name = bm
        best_version = best_models[bm]
        model_path = os.path.join(MODELS_PATH[model_name], best_version)
        model = torch.load(model_path)
        model.eval()
        model.to(DEVICE)
        print(f"Evaluating best model {model_name} version {best_version} on test set")
        if "TL" in model_name:
            test_dataset = test_dataset_TL
        else:
            test_dataset = test_dataset_base

        TEST_RESULTS[model_name] = evaluate_model(model, test_dataset)
        print("")

    with open('output/test_results.json', 'w') as f:
        json.dump(TEST_RESULTS, f, indent=4)

    create_leaderboard_test()