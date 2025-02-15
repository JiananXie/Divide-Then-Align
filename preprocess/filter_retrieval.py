import json
import os
from tqdm import tqdm
from collections import Counter
from preprocess.vllm_inference import vllm_wo_retrieval,vllm_w_retrieval,extract_answer
from metrics.em_f1 import compute_metrics

def inference_wo_retrieval(args,data_path,save_path,data_type):
    data = open(data_path,encoding='utf-8')
    data = json.load(data)
    k_data = []
    uk_data = []

    batch_samples = []
    for sample_idx, sample in enumerate(data):
        batch_samples.append({
            'question': sample['question'],
            'sample_idx': sample_idx,
        })
    
    texts = vllm_wo_retrieval(args, batch_samples)
    
    k_counts = [0] * len(data)
    correct_list = [[] for _ in range(len(data))]
    wrong_list = [[] for _ in range(len(data))]
    
    for i, generated_k_texts in enumerate(texts):
        sample_idx = batch_samples[i]['sample_idx']
        for text in generated_k_texts:
            text = extract_answer(args,text)
            if text:
                output = compute_metrics(text, data[sample_idx]['answers'])
                f1 = output['f1']
                em = output['em']
            
                if em == 1:
                    k_counts[sample_idx] += 1
                    correct_list[sample_idx].append(text.strip())
                else:
                    wrong_list[sample_idx].append(text.strip())
    if data_type == "train":
        for i, sample in tqdm(enumerate(data), desc=f"Filtering {data_type} questions"):
            if k_counts[i] >= args.infer_k * 1.0 or (correct_list[i] and not wrong_list[i]):
                sample['answer'] = Counter(correct_list[i]).most_common(1)[0][0]
                k_data.append(sample)
            elif wrong_list[i]:
                sample['answer'] = Counter(wrong_list[i]).most_common(1)[0][0]
                uk_data.append(sample)
    else:
        for i, sample in tqdm(enumerate(data), desc=f"Filtering {data_type} questions"):
            if k_counts[i] >= args.infer_k * 1.0:
                k_data.append(sample)
            else:
                uk_data.append(sample)

    try:
        print("known data:",len(k_data))
        print("unknown data:",len(uk_data))
        with open(os.path.join(save_path,'known.json'),'w',encoding='utf-8') as f:
            json.dump(k_data,f,ensure_ascii=False,indent=2)
        with open(os.path.join(save_path,'unknown.json'),'w',encoding='utf-8') as f:
            json.dump(uk_data,f,ensure_ascii=False,indent=2)
    except IOError as e:
        print(f"Error writing to file: {e}")

def inference_w_retrieval(args,data_path,save_path,data_type):
    data = open(data_path,encoding='utf-8')
    data = json.load(data)
    
    texts = vllm_w_retrieval(args, data)

    em_counts = [0] * len(data)
    correct_list = [[] for _ in range(len(data))]
    wrong_list = [[] for _ in range(len(data))]

    for i, generated_k_texts in enumerate(texts):
        for text in generated_k_texts:
            text = extract_answer(args,text)
            if text:
                output = compute_metrics(text, data[i]['answers'])
                em = output['em']
                em_counts[i] += em
                if em == 1:
                    correct_list[i].append(text.strip())
                else:
                    wrong_list[i].append(text.strip())
    
    filter_data = []
    for i, sample in tqdm(enumerate(data), desc=f"Filtering {data_type} questions with golden retrieval"):
        sample['em'] = em_counts[i]
        if not correct_list[i] and not wrong_list[i]:
            continue
        if correct_list[i]:
            sample['ctx_answer'] = Counter(correct_list[i]).most_common(1)[0][0]
        else:
            sample['ctx_answer'] = ""
        if wrong_list[i]:
            sample['ctx_wrong_answer'] = Counter(wrong_list[i]).most_common(1)[0][0]
        else:
            sample['ctx_wrong_answer'] = ""
        filter_data.append(sample)
    try:
        print(f"{data_type} data after filtering retrieval:",len(filter_data))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error writing to file: {e}")