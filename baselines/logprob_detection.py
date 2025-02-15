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
    """计算logprob检测的评估指标
    Args:
        scores: logprob分数列表
        data: 原始数据列表，包含answer, answers, type等字段
        threshold: logprob的阈值
    Returns:
        dict: 包含各项指标的字典
    """
    # 初始化结果统计
    result_dict = {
        'kg': {'tt': 0, 'oo': 0},
        'kn': {'tt': 0, 'oo': 0}, 
        'ug': {'tt': 0, 'oo': 0},
        'un': {'tt': 0, 'oo': 0}
    }
    
    # 统计各类型数量
    len_kg = sum(1 for item in data if item['type'] == 'kg')
    len_kn = sum(1 for item in data if item['type'] == 'kn')
    len_ug = sum(1 for item in data if item['type'] == 'ug')
    len_un = sum(1 for item in data if item['type'] == 'un')
    
    # 计算每个预测的指标
    for score, item in zip(scores, data):
        if score >= threshold:  # 模型选择回答
            # 检查答案是否正确
            output = compute_metrics(item['prediction'], item['answers'])
            em = output['em']
            # 增加tt
            result_dict[item['type']]['tt'] += em
        else:  # 模型选择拒答
            # 增加oo
            result_dict[item['type']]['oo'] += 1
    
    # 计算最终指标
    total = len_kg + len_kn + len_ug + len_un
    if total == 0:
        return {
            'metrics': {
                "Accuracy": 0.0,
                "Recall": 0.0,
                "Precision": 0.0,
                "Denoise rate": 0.0,
                "Context utilization rate": 0.0,
                "Hullucination rate": 0.0,
                "Abstain Recall": 0.0,
                "Abstain Precision": 0.0
            }
        }
    
    acc = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
           result_dict['ug']['tt'] + result_dict['un']['oo']) / total
    
    # 计算precision时检查分母是否为0
    precision_denominator = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
                           result_dict['ug']['tt'] + result_dict['un']['tt'])
    precision = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
               precision_denominator if precision_denominator > 0 else 0.0
    
    # 计算recall时检查分母是否为0
    recall_denominator = (len_kg + len_kn + len_ug)
    recall = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
             recall_denominator if recall_denominator > 0 else 0.0
    
    # 其他指标也添加防护
    denoise = result_dict['kn']['tt'] / len_kn if len_kn > 0 else 0.0
    ctx_util = result_dict['ug']['tt'] / len_ug if len_ug > 0 else 0.0
    hullucination = 1 - (result_dict['un']['tt'] + result_dict['un']['oo']) / len_un if len_un > 0 else 0.0
    abstain_recall = result_dict['un']['oo'] / len_un if len_un > 0 else 0.0
    
    total_oo = sum(result_dict[k]['oo'] for k in ['kg', 'kn', 'ug', 'un'])
    abstain_precision = result_dict['un']['oo'] / total_oo if total_oo > 0 else 0.0
    
    metrics = {
        "Accuracy": acc,
        "Recall": recall,
        "Precision": precision,
        "Denoise rate": denoise,
        "Context utilization rate": ctx_util,
        "Hullucination rate": hullucination,
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
    
    # 使用vLLM生成预测，设置return_logprobs=True
    outputs = vllm_w_retrieval(args, data, return_logprobs=True)
    
    # 一次循环处理所有数据
    for item, output in zip(data, outputs):
        # 提取logprobs和probabilities
        logprobs = output['logits']
        probs = output['probabilities']
        
        if not logprobs:
            # 初始化所有分数为-inf
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
            # 计算所有统计指标
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
        
        # 保存所有分数
        item['scores'] = scores
        # 根据指定的metric选择分数
        item['logprob_score'] = scores[f'logprob_{args.metric}']
        # 提取并保存预测文本
        item['prediction'] = extract_answer(args, output['text'])
    
    # 计算评估指标
    result_dict = evaluate_predictions(
        scores=[item['logprob_score'] for item in data],
        data=data,
        threshold=args.threshold
    )
    
    # 保存结果
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path)
    result_file = os.path.join(
        args.result_path, 
        f'logprob_results_{model_name}_t{args.threshold}_{args.metric}_ctx{args.ctxs_num}.json'
    )
    
    results = {
        'result_dict': result_dict,
        'data': data,  # 现在包含了每个样本的prediction、所有分数和选用的logprob_score
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