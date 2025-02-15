import torch
import argparse
import numpy as np
import os
import sys
import json
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm
from preprocess.vllm_inference import extract_answer
from evaluations import compute_evaluation_metrics
from transformers import AutoTokenizer
def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect hallucinations using ICL examples')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="Result",
                       help="Path to save results")
    parser.add_argument("--ctxs_num", type=int, default=3,
                       help="Number of contexts to use")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of ICL examples to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint for RAAT model")
    args = parser.parse_args()
    return args

def get_icl_examples(model_name, data, num_examples=3, seed=42):
    """Get in-context learning examples including both answer and refusal cases"""
    random.seed(seed)
    
    # Get one refusal example from 'un' type
    refusal_examples = [x for x in data if x['type'] == 'un']
    refusal_example = random.choice(refusal_examples)
    
    # Get answer examples from other types
    answer_examples = [x for x in data if x['type'] in ['kg', 'kn', 'ug']]
    answer_examples = random.sample(answer_examples, num_examples - 1)
    
    # Combine and shuffle examples
    examples = answer_examples + [refusal_example]
    random.shuffle(examples)
    
    formatted_examples = ""
    refusal_answer = "This question is beyond the scope of my knowledge and the references, I don't know the answer."
    sys_ctxs = 'The following contexts will help you complete the question-and-answer pair.'
    if 'chatqa' not in model_name.lower():
        for ex in examples:
            if len(ex['ctxs']) >= 1:
                formatted_examples += f"\nQuestion: {ex['question']}"
                # Add contexts
                if len(ex['ctxs']) == 1:
                    formatted_examples += f"\nContext:{ex['ctxs'][0]['text']}"
                else:
                    for i, ctx in enumerate(ex['ctxs'], 1):
                        formatted_examples += f"\nContext{i}:{ctx['text']}"
                # Add answer
                formatted_examples += f"\nAnswer: {refusal_answer if ex['type'] == 'un' else ex['answers'][0]}."
        return formatted_examples
    else:
        icl_pairs = []
        for ex in examples:
            formatted_examples = ""
            if len(ex['ctxs']) >= 1:
                formatted_examples += f"Question: {ex['question']}"
                # Add contexts
                if len(ex['ctxs']) == 1:
                    formatted_examples += f"\nContext:{ex['ctxs'][0]['text']}"
                else:
                    for i, ctx in enumerate(ex['ctxs'], 1):
                        formatted_examples += f"\nContext{i}:{ctx['text']}"
                # Add answer
                icl_pairs.append((formatted_examples, f"{refusal_answer if ex['type'] == 'un' else ex['answers'][0]}."))
                
        return icl_pairs

def format_prompt(question, contexts, icl_examples, model_name, args):
    """Format prompt according to model type as in vllm_evaluation.py"""
    sys = "You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences. If you don't know the answer and the following contexts do not contain the necessay information to answer the question, respond with 'This question is beyond the scope of my knowledge and the references, I don't know the answer'."
    sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'
    sys_ctxs = 'The following contexts will help you complete the question-and-answer pair.'
    
    # Format contexts
    if len(contexts) == 1:
        ctx_text = f"{sys_ctx}{contexts[0]['text']}"
    else:
        ctx_text = f"{sys_ctxs}"
        for i, ctx in enumerate(contexts, 1):
            ctx_text += f"\nContext{i}:{ctx['text']}"
    # Add examples to system prompt
    sys_with_examples = sys + icl_examples
    # Format according to model type
    if 'retrobust' in model_name:
        prompt = f"{sys_with_examples}\n{ctx_text}\nQuestion:{question}\nAre follow up questions needed here: No.\nSo the final answer is:"
    elif 'llama2' in model_name.lower() or 'raat' in model_name:
        if args.checkpoint_path:
            prompt = f"<<SYS>>\n{sys_with_examples}\n<</SYS>>\n\n[INST] {ctx_text}\nQuestion:{question} [/INST]"
        else:
            prompt = f"{sys_with_examples}\n\n{ctx_text}\nQuestion:{question}"
    elif 'chatqa' in model_name.lower():
        prompt_format='{system_prompt}\n\n{ctx}\n{query}'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{sys}\n\n{ex[0]}"} for ex in icl_examples for _ in (0,)] +
            [{"role": "assistant", "content": f"{ex[1]}"} for ex in icl_examples for _ in (0,)] +
            [{"role": "user", "content": prompt_format.format(system_prompt=sys,ctx=ctx_text,query='Question:'+question)}],
            tokenize=False,
            add_generation_prompt=True)
    else:  # Default format
        prompt = f"{sys_with_examples}\n{ctx_text}\n\nQuestion:{question}"
    
    return prompt

def evaluate_with_icl(args, data):
    """Evaluate using in-context learning approach"""
    # Get ICL examples
    icl_examples = get_icl_examples(args.model_name_or_path, data, args.num_examples, args.seed)
    # Prepare prompts for each question
    prompts = []
    all_records = [] 
    
    for item in tqdm(data):
        prompt = format_prompt(
            item['question'],
            item.get('ctxs', []),
            icl_examples,
            args.model_name_or_path,
            args
        )
        prompts.append(prompt)
        record = {
            'question': item['question'],
            'prompt': prompt,
            'contexts': item.get('ctxs', []),
            'type': item['type'],
            'ground_truth': item['answers']
        }
        all_records.append(record)
    
    print('prompt done')
    # Initialize sampling parameters to match vllm_evaluation.py
    sampling_params = SamplingParams(
        max_tokens=512,
        top_k=50,
        top_p=0.6,
        temperature=0.3,
        repetition_penalty=1,
        stop_token_ids=[7],
    )
    
    # Adjust parameters for specific models
    if 'retrobust' in args.model_name_or_path:
        sampling_params = SamplingParams(
            max_tokens=512,
            top_k=50,
            top_p=0.6,
            temperature=0.95,
            repetition_penalty=1.0,
            stop=['#']
        )
    
    # Initialize vLLM with checkpoint support
    llm = LLM(
        model=args.model_name_or_path, 
        tensor_parallel_size=1,
        tokenizer_mode='auto',
        trust_remote_code=True,
        load_format='auto',
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=16384
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path:    
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())
    
    # Generate answers - vLLM handles batching internally
    results = llm.generate(prompts, sampling_params)
    outputs = [output.outputs[0].text for output in results]
    

    for record, output in zip(all_records, outputs):
        record['model_output'] = output
        record['extracted_answer'] = extract_answer(args, output)
    
    return outputs, all_records 

def main():
    args = parse_args()
    
    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get model outputs and records
    outputs, all_records = evaluate_with_icl(args, data) 
    
    # Extract answers and update data
    for item, output in zip(data, outputs):
        item['prediction'] = extract_answer(args, output)
    
    # Prepare data for metrics computation
    predictions = [item['prediction'] for item in data]
    answers = [item['answers'] for item in data]
    types = [item['type'] for item in data]
    refusal_answer = "This question is beyond the scope of my knowledge and the references, I don't know the answer."
    
    # Compute evaluation metrics using the same function as in evaluations.py
    result_dict = compute_evaluation_metrics(predictions, answers, types, refusal_answer)
    
    # Save results
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path)
    result_file = os.path.join(
        args.result_path, 
        f'icl_results_{model_name}_ex{args.num_examples}_ctx{args.ctxs_num}.json'
    )
    
    results = {
        'data': data,
        'num_examples': args.num_examples,
        'inference_records': all_records,  
        'metrics': result_dict['metrics']  
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print("\nMetrics:")
    for metric, value in result_dict['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 