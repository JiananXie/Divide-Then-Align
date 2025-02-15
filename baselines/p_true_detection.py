import torch
import argparse
import numpy as np
import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from metrics.em_f1 import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect hallucinations using p_true scores')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for p_true score to determine refusal")
    parser.add_argument("--reversed", action="store_true",
                       help="Whether to reverse the order of A/B tokens")
    parser.add_argument("--result_path", type=str, default="Result",
                       help="Path to save results")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint for RAAT model")
    args = parser.parse_args()
    return args

def get_p_true_score(llm, tokenizer, question, prediction, reversed=False):
    """Get the model's probability judgment of answer correctness
    using the logits distribution of the first token across the entire vocabulary
    """
    prompt = f"""Question: {question}
Possible answers: {prediction}
Is the possible answer:
(A) True
(B) False
The possible answer is:"""

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=tokenizer.vocab_size,  # Request logprobs for entire vocabulary
        stop=None
    )
    
    # Get A and B token ids
    token_a = tokenizer.encode('A', add_special_tokens=False)[0]
    token_b = tokenizer.encode('B', add_special_tokens=False)[0]
    
    outputs = llm.generate([prompt], sampling_params)
    output = outputs[0]
    
    # Get logprobs for the entire vocabulary
    logprobs_dict = output.outputs[0].logprobs[0]
    
    # Get logprobs for A and B tokens
    logit_a = logprobs_dict[token_a].logprob
    logit_b = logprobs_dict[token_b].logprob
    
    # Calculate softmax probabilities
    logits = torch.tensor([logit_a, logit_b])
    probs = torch.softmax(logits, dim=0)
    
    # Return probability based on reversed parameter
    return probs[1].item() if reversed else probs[0].item()

def compute_p_true_metrics(scores, data, threshold=0.5):
    """Calculate evaluation metrics for p_true detection
    Args:
        scores: list of p_true scores
        data: original data list containing answer, answers, type fields
        threshold: threshold for p_true
    Returns:
        dict: dictionary containing various metrics
    """
    # Initialize result statistics
    result_dict = {
        'kg': {'tt': 0, 'oo': 0},
        'kn': {'tt': 0, 'oo': 0}, 
        'ug': {'tt': 0, 'oo': 0},
        'un': {'tt': 0, 'oo': 0}
    }
    
    # Count numbers for each type
    len_kg = sum(1 for item in data if item['type'] == 'kg')
    len_kn = sum(1 for item in data if item['type'] == 'kn')
    len_ug = sum(1 for item in data if item['type'] == 'ug')
    len_un = sum(1 for item in data if item['type'] == 'un')
    
    # Calculate metrics for each prediction
    for score, item in zip(scores, data):
        if score > threshold: 
            output = compute_metrics(item['prediction'], item['answers'])
            em = output['em']
            result_dict[item['type']]['tt'] += em
        else: 
            result_dict[item['type']]['oo'] += 1
    
    # Calculate final metrics
    acc = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
           result_dict['ug']['tt'] + result_dict['un']['oo']) / (len_kg + len_kn + len_ug + len_un)
    
    total_predicted_positive = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
                              result_dict['ug']['tt'] + result_dict['un']['tt'])
    precision = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
                total_predicted_positive if total_predicted_positive > 0 else float('nan')
    
    total_actual_positive = (len_kg + len_kn + len_ug)
    recall = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
             total_actual_positive if total_actual_positive > 0 else float('nan')
    
    denoise = result_dict['kn']['tt'] / len_kn if len_kn > 0 else 0
    ctx_util = result_dict['ug']['tt'] / len_ug if len_ug > 0 else 0
    hallucination = 1 - (result_dict['un']['tt'] + result_dict['un']['oo']) / len_un if len_un > 0 else 0
    abstain_recall = result_dict['un']['oo'] / len_un if len_un > 0 else 0
    
    total_oo = sum(result_dict[k]['oo'] for k in ['kg', 'kn', 'ug', 'un'])
    abstain_precision = result_dict['un']['oo'] / total_oo if total_oo > 0 else float('nan')
    
    metrics = {
        "Accuracy": acc,
        "Recall": recall,
        "Precision": precision,
        "Denoise rate": denoise,
        "Context utilization rate": ctx_util,
        "Hallucination rate": hallucination,
        "Abstain Recall": abstain_recall,
        "Abstain Precision": abstain_precision
    }
    result_dict['metrics'] = metrics
    return result_dict

def main():
    args = parse_args()
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(model=args.model_name_or_path,
            tensor_parallel_size=1,
            trust_remote_code=True, max_logprobs=tokenizer.vocab_size)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = [item for item in raw_data if isinstance(item, dict) and 
            all(key in item for key in ['question', 'prediction', 'type'])]
    
    all_scores = []
    
    for item in tqdm(data):
        score = get_p_true_score(llm, tokenizer, item['question'], item['prediction'], args.reversed)
        all_scores.append(score)
        item['p_true_score'] = score
    
    # Calculate evaluation metrics
    result_dicts = compute_p_true_metrics(all_scores, data, args.threshold)
    
    # Save results
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path)
    result_file = os.path.join(
        args.result_path, 
        f'p_true_results_{model_name}_t{args.threshold}_{"rev" if args.reversed else "std"}.json'
    )
    
    results = {
        'result_dicts': result_dicts,
        'data': data, 
        'threshold': args.threshold,
        'reversed': args.reversed
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print("\nMetrics:")
    for metric, value in result_dicts['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 