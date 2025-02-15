import torch
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from metrics.em_f1 import compute_metrics
from preprocess.vllm_inference import vllm_w_retrieval, extract_answer
from evaluations import compute_evaluation_metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect hallucinations using self-consistency')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold for consistency score to determine refusal")
    parser.add_argument("--infer_k", type=int, default=10,
                       help="Number of samples to generate for each question")
    parser.add_argument("--result_path", type=str, default="Result",
                       help="Path to save results")
    parser.add_argument("--ctxs_num", type=int, default=1,
                       help="Number of contexts to use")
    parser.add_argument("--checkpoint_path", type=str, default=None,)
    args = parser.parse_args()

    return args

def compute_consistency_score(predictions):
    """Compute consistency score based on multiple predictions
    
    Args:
        predictions: List of predictions for the same question
        
    Returns:
        float: Consistency score between 0 and 1
    """
    if not predictions:
        return 0.0
        
    # Count exact matches between all pairs of predictions
    n = len(predictions)
    if n <= 1:
        return 0.0
        
    match_count = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Use exact match score between predictions
            match = compute_metrics(predictions[i], [predictions[j]])['em']
            match_count += match
            total_pairs += 1
            
    # Return average consistency score
    return match_count / total_pairs if total_pairs > 0 else 0.0

def main():
    args = parse_args()
    
    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Generate multiple answers using vllm_w_retrieval
    outputs = vllm_w_retrieval(args, data, "self_consistency")
    
    # Process outputs and compute consistency scores
    all_predictions = []
    consistency_scores = []
    
    for i, sample_outputs in enumerate(outputs):
        # Extract answers from outputs
        predictions = [extract_answer(args, output) for output in sample_outputs]
        predictions = [p for p in predictions if p]  # Remove False/None values
        
        # Compute consistency score
        consistency_score = compute_consistency_score(predictions)
        consistency_scores.append(consistency_score)
        
        # Use most common prediction as final prediction
        if predictions:
            # Count occurrences of each prediction
            from collections import Counter
            prediction_counts = Counter(predictions)
            most_common = prediction_counts.most_common(1)[0][0]
            all_predictions.append(most_common)
        else:
            all_predictions.append("")
            
        # Store scores and predictions in data
        data[i]['consistency_score'] = consistency_score
        data[i]['multiple_predictions'] = predictions
        data[i]['prediction'] = most_common
    
    # Determine final predictions based on threshold
    refusal_answer = "This question is beyond the scope of my knowledge and the references, I don't know the answer."
    final_predictions = []
    
    for score, pred in zip(consistency_scores, all_predictions):
        if score >= args.threshold:
            final_predictions.append(pred)
        else:
            final_predictions.append(refusal_answer)
    
    # Compute evaluation metrics
    result_dict = compute_evaluation_metrics(
        final_predictions,
        [item['answers'] for item in data],
        [item['type'] for item in data],
        refusal_answer
    )
    
    # Save results
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path)
    result_file = os.path.join(
        args.result_path,
        f'consistency_results_{model_name}_n{args.infer_k}_t{args.threshold}_ctx{args.ctxs_num}.json'
    )
    
    results = {
        'result_dict': result_dict,
        'data': data, 
        'threshold': args.threshold,
        'num_samples': args.infer_k
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print("\nMetrics:")
    for metric, value in result_dict['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 