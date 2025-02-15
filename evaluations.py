import os
import json
import random
from metrics.em_f1 import compute_metrics
from preprocess.vllm_inference import extract_answer
from preprocess.vllm_evaluation import vllm_w_retrieval, merge_and_save_peft_model
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import HfArgumentParser

# try:
#     multiprocessing.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass  

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the test model name or path"},
    )
    checkpoint_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the checkpoint path"},
    )
    benchmark: Optional[str] = field(
        default = "knowledge",
        metadata={
            "help": "benchmark to evaluate model",
            "choices": ["knowledge", "crag","rgb"]
        },
    )
    data_path: Optional[str] = field(
        default="data_kbrag",
        metadata={"help": "the location of the data path"},
    )
    cache_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the cache path"},
    )
    datasets: Optional[str] = field(
        default="nq,triviaqa,webq",
        metadata={"help": "the datasets to be processed in the format of 'dataset1,dataset2,dataset3'"},
    )
    result_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the result path"},
    )
    ctxs_num: Optional[int] = field(
        default=3,
        metadata={"help": "the number of ctxs to be used"},
    )
    total_size: Optional[int] = field(
        default=3000,
        metadata={"help": "the number of data to be used"},
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "random seed for sampling"},
    )

def compute_evaluation_metrics(predictions, answers, types, refusal_answer="NO ANSWER"):
    """Compute evaluation metrics
    Args:
        predictions: predictions from model
        answers: ground truth answers
        types: data types ('kg', 'kn', 'ug', 'un')
        refusal_answer: refusal answer template(default: "NO ANSWER")
    Returns:
        dict: dictionary containing all metrics
    """
    # initialize result statistics
    result_dict = {
        'kg': {'tt': 0, 'oo': 0},
        'kn': {'tt': 0, 'oo': 0}, 
        'ug': {'tt': 0, 'oo': 0},
        'un': {'tt': 0, 'oo': 0}
    }
    
    # count the number of each type
    len_kg = sum(1 for t in types if t == 'kg')
    len_kn = sum(1 for t in types if t == 'kn')
    len_ug = sum(1 for t in types if t == 'ug')
    len_un = sum(1 for t in types if t == 'un')
    
    # compute the metrics of each prediction
    for i, (pred, ans, item_type) in enumerate(zip(predictions, answers, types)):
        if pred:
            # compute the match between prediction and ground truth
            output = compute_metrics(pred, ans)
            em = output['em']
            # compute whether it is a refusal answer
            oo = compute_metrics(pred, [refusal_answer])['em']
        else:
            em = 0
            oo = 0
            
        # accumulate the statistics of each type
        result_dict[item_type]['tt'] += em
        result_dict[item_type]['oo'] += oo
    
    # compute the final metrics
    acc = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + 
           result_dict['ug']['tt'] + result_dict['un']['oo']) / (len_kg + len_kn + len_ug + len_un)
    
    precision = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
                (len_kg + len_kn + len_ug + len_un - result_dict['un']['oo'] - result_dict['ug']['oo'] - result_dict['kn']['oo'] - result_dict['kg']['oo'])
    
    recall = (result_dict['kg']['tt'] + result_dict['kn']['tt'] + result_dict['ug']['tt']) / \
             (len_kg + len_kn + len_ug)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    denoise = result_dict['kn']['tt'] / len_kn if len_kn > 0 else 0
    ctx_util = result_dict['ug']['tt'] / len_ug if len_ug > 0 else 0
    hallucination = 1 - (result_dict['un']['tt'] + result_dict['un']['oo']) / len_un if len_un > 0 else 0
    abstain_recall = result_dict['un']['oo'] / len_un if len_un > 0 else 0
    
    total_oo = sum(result_dict[k]['oo'] for k in ['kg', 'kn', 'ug', 'un'])
    abstain_precision = result_dict['un']['oo'] / total_oo if total_oo > 0 else float('nan')
    abstain_f1 = 2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall) if abstain_precision + abstain_recall > 0 and abstain_precision != float('nan') and abstain_recall != float('nan') else 0
    metrics = {
        "Accuracy": acc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Denoise rate": denoise,
        "Context utilization rate": ctx_util,
        "Hullucination rate": hallucination,
        "Abstain Recall": abstain_recall,
        "Abstain Precision": abstain_precision,
        "Abstain F1": abstain_f1
    }
    result_dict['metrics'] = metrics
    return result_dict

def evaluate_knowledge(result_model_name, data_model_name, datasets, args):
    random.seed(args.seed)
    refusal_answer = "This question is beyond the scope of my knowledge and the references, I don't know the answer."
    if not args.cache_path:
        raise ValueError("cache_path is required to load cached dataset or save evaluation data")
    
    if os.path.exists(args.cache_path):
        with open(args.cache_path,'r',encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Evaluation data loaded from {args.cache_path}")
    else:
        all_data = []
        for dataset in datasets:
            data_path = os.path.join(args.data_path,data_model_name,dataset,'eval_data.json')
            with open(data_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            print(len(data))
            all_data.extend(data)
        random.shuffle(all_data)


        kg_data = [e for e in all_data if e['type'] == 'kg']
        kn_data = [e for e in all_data if e['type'] == 'kn']
        un_data = [e for e in all_data if e['type'] == 'un']
        ug_data = [e for e in all_data if e['type'] == 'ug']
        print(f"kg,kn,ug,un: {len(kg_data)},{len(kn_data)},{len(ug_data)},{len(un_data)}")

        test_data = random.sample(all_data,args.total_size)
        kg_data = [e for e in test_data if e['type'] == 'kg']
        kn_data = [e for e in test_data if e['type'] == 'kn']
        ug_data = [e for e in test_data if e['type'] == 'ug']
        un_data = [e for e in test_data if e['type'] == 'un']
        with open(args.cache_path,'w',encoding='utf-8') as f:
            json.dump(test_data,f,ensure_ascii=False,indent=2)
        print(f'''Evaluation data generated for {data_model_name} has {len(kg_data)+len(kn_data)+len(un_data)+len(ug_data)} samples.
        Known-Golden: {len(kg_data)} samples.
        Known-Noisy: {len(kn_data)} samples.
        Unknown-Golden: {len(ug_data)} samples.
        Unknown-Noisy: {len(un_data)} samples.''')

    texts = vllm_w_retrieval(args, test_data)
    predictions = [extract_answer(args, text) for text in texts]
    answers = [item['answers'] for item in test_data]
    types = [item['type'] for item in test_data]
    
    result_dict = compute_evaluation_metrics(predictions, answers, types, refusal_answer)
    

    for i, (pred, text) in enumerate(zip(predictions, texts)):
        test_data[i]['prediction'] = pred.strip() if pred else ""
    test_data.append(result_dict)
    
    os.makedirs(os.path.join(args.result_path, "knowledge"), exist_ok=True)
    with open(os.path.join(args.result_path, "knowledge", f'{args.ctxs_num}_{result_model_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args()
    # Handle PEFT model if necessary
    if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
        print("PEFT model detected, merging with base model...")
        args.model_name_or_path = merge_and_save_peft_model(args.model_name_or_path)
    
    result_model_name = args.model_name_or_path.split('/')[-1]  # For result storage
    data_model_name = result_model_name  # For data loading
    
    if "retrobust" in data_model_name:
        data_model_name = "Llama-2-13b-hf"
        result_model_name = args.model_name_or_path.split('/')[-1]  # For result storage
    if data_model_name == "final_checkpoint" or data_model_name == "best_checkpoint":
        data_model_name = args.model_name_or_path.split('/')[-2]
        result_model_name = data_model_name
    if args.checkpoint_path:
        data_model_name = args.checkpoint_path.split('/')[-2]
        result_model_name = data_model_name  # Update result_model_name for PEFT models

    datasets = [e.strip() for e in args.datasets.split(',')]

    if args.benchmark == "knowledge":
        evaluate_knowledge(result_model_name, data_model_name, datasets, args)