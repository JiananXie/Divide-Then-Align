import argparse
import os
import json
from metrics.em_f1 import compute_metrics
from preprocess.vllm_evaluation import vllm_w_retrieval
from preprocess.vllm_inference import extract_answer

def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect hallucinations using logprob scores')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for logprob score to determine refusal")
    parser.add_argument("--metric", choices=['min', 'mean', 'last_token'], 
                       default='mean',
                       help="Which logprob metric to use")
    parser.add_argument("--result_path", type=str, default="Result",
                       help="Path to save results")
    parser.add_argument("--ctxs_num", type=int, default=1,
                       help="Number of contexts to use")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to save checkpoint")
    args = parser.parse_args()
    return args

def evaluate_predictions(scores, data, threshold=0.5):
    """Calculate evaluation metrics for logprob detection
    Args:
        scores: list of logprob scores
        data: original data list containing answer, answers, type fields
        threshold: threshold for logprob
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
        if score >= threshold: 
            output = compute_metrics(item['prediction'], item['answers'])
            em = output['em']
            result_dict[item['type']]['tt'] += em
        else: 
            result_dict[item['type']]['oo'] += 1
    
    # Calculate final metrics
    total = len_kg + len_kn + len_ug + len_un
    if total == 0:
        return {
            'metrics': {
                "Accuracy": 0.0,
                "Recall": 0.0,
                "Precision": 0.0,
                "Denoise rate": 0.0,
                "Context utilization rate": 0.0,
                "Hallucination rate": 0.0,
                "Abstain Recall": 0.0,
                "Abstain Precision": 0.0
            }
        }
    
    acc = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
           result_dict['ug']['tt'] + result_dict['un']['oo']) / total
    
    precision_denominator = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
                           result_dict['ug']['tt'] + result_dict['un']['tt'])
    precision = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
               precision_denominator if precision_denominator > 0 else 0.0
    
    recall_denominator = (len_kg + len_kn + len_ug)
    recall = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
             recall_denominator if recall_denominator > 0 else 0.0
    
    denoise = result_dict['kn']['tt'] / len_kn if len_kn > 0 else 0.0
    ctx_util = result_dict['ug']['tt'] / len_ug if len_ug > 0 else 0.0
    hallucination = 1 - (result_dict['un']['tt'] + result_dict['un']['oo']) / len_un if len_un > 0 else 0.0
    abstain_recall = result_dict['un']['oo'] / len_un if len_un > 0 else 0.0
    
    total_oo = sum(result_dict[k]['oo'] for k in ['kg', 'kn', 'ug', 'un'])
    abstain_precision = result_dict['un']['oo'] / total_oo if total_oo > 0 else 0.0
    
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
    
    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate predictions using vLLM with return_logprobs=True
    outputs = vllm_w_retrieval(args, data, return_logprobs=True)
    
    for item, output in zip(data, outputs):
        logprobs = output['logits']
        probs = output['probabilities']
        
        if not logprobs:
            scores = {
                'logprob_min': float('-inf'),
                'logprob_max': float('-inf'),
                'logprob_mean': float('-inf'),
                'logprob_last': float('-inf'),
                'prob_min': 0.0,
                'prob_max': 0.0,
                'prob_mean': 0.0,
                'prob_last': 0.0
            }
        else:
            scores = {
                'logprob_min': min(logprobs),
                'logprob_max': max(logprobs),
                'logprob_mean': sum(logprobs) / len(logprobs),
                'logprob_last': logprobs[-1],
                'prob_min': min(probs),
                'prob_max': max(probs),
                'prob_mean': sum(probs) / len(probs),
                'prob_last': probs[-1]
            }
        
        item['scores'] = scores
        item['logprob_score'] = scores[f'logprob_{args.metric}']
        item['prediction'] = extract_answer(args, output['text'])
    
    # Calculate evaluation metrics
    result_dict = evaluate_predictions(
        scores=[item['logprob_score'] for item in data],
        data=data,
        threshold=args.threshold
    )
    
    # Save results
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path)
    result_file = os.path.join(
        args.result_path, 
        f'logprob_results_{model_name}_t{args.threshold}_{args.metric}_ctx{args.ctxs_num}.json'
    )
    
    results = {
        'result_dict': result_dict,
        'data': data, 
        'threshold': args.threshold,
        'metric': args.metric
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print("\nMetrics:")
    for metric, value in result_dict['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 