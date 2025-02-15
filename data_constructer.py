import os
from preprocess.filter_ctx import *
from preprocess.filter_retrieval import *
from preprocess.vllm_inference import *
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import HfArgumentParser

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    datasets: Optional[str] = field(
        default="nq,triviaqa,webq",
        metadata={"help": "the datasets to be processed in the format of 'dataset1,dataset2,dataset3'"},
    )
    data_path: Optional[str] = field(
        default="data",
        metadata={"help": "the location of the data path"},
    )
    infer_k: Optional[int] = field(
        default=10,
        metadata={"help": "inference times of per query"},
    )
    checkpoint_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the checkpoint path"},
    )
    eval: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to generate evaluation data"},
    )



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name = script_args.model_name_or_path.split('/')[-1]
    datasets = [e.strip() for e in script_args.datasets.split(',')]
    eval = script_args.eval

    for source in datasets: 
        print(f"Processing {source} dataset...")
        data_type = "test" if eval else "train"
        # Expore the boundary of the model knowledge
        data_path = os.path.join(script_args.data_path,source,data_type+'.json')
        save_path = os.path.join(script_args.data_path,model_name,source,data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        inference_wo_retrieval(script_args,data_path=data_path,save_path=save_path,data_type=data_type)

        refusal_answer = "This question is beyond the scope of my knowledge and the references, I don't know the answer."
        if not eval:    
            #Generate DPO training data(train+valid)
            data_path = os.path.join(script_args.data_path,model_name,source)
            unknown_path = os.path.join(data_path,'train','unknown.json')
            known_path = os.path.join(data_path,'train','known.json')
            
            with open(unknown_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            uk_data = []
            for item in data:
                ctxs_pool = item['possible_golden_ctxs'] + item['possible_noisy_ctxs']
                if len(ctxs_pool) < 3:
                    continue
                sorted_ctxs_pool = sorted(ctxs_pool, key=lambda x: x['score'], reverse=True)
                ctxs_pool = sorted_ctxs_pool[:3]
                g_count = 0
                for ctx in ctxs_pool:
                    if ctx['type'] == 'g':
                        g_count += 1
                if g_count > 0:
                    #ug: unknown question with golden retrieval
                    uk_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "answer": item['answer'],
                        "type": "ug"
                    })
                else:
                    #un: unknown question with noisy retrieval
                    uk_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "answer": item['answer'],
                        "type": "un"
                    })
            with open(unknown_path,'w',encoding='utf-8') as f:
                json.dump(uk_data,f,ensure_ascii=False,indent=2)
# 
            with open(known_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            k_data = []
            for item in data:
                ctxs_pool = item['possible_golden_ctxs'] + item['possible_noisy_ctxs']
                if len(ctxs_pool) < 3:
                    continue
                sorted_ctxs_pool = sorted(ctxs_pool, key=lambda x: x['score'], reverse=True)
                ctxs_pool = sorted_ctxs_pool[:3]
                g_count = 0
                for ctx in ctxs_pool:
                    if ctx['type'] == 'g':
                        g_count += 1
                if g_count > 0:
                    #kg: known question with golden retrieval
                    k_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "answer": item['answer'],
                        "type": "kg"
                    })
                else:
                    #kn: known question with noisy retrieval
                    k_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "answer": item['answer'],
                        "type": "kn"
                    })
            with open(known_path,'w',encoding='utf-8') as f:
                json.dump(k_data,f,ensure_ascii=False,indent=2)
# 
            # #Filter the unknown questions with retrieval
            inference_w_retrieval(script_args,data_path=unknown_path,save_path=unknown_path,data_type="unknown")
    # 
            # #Filter the known questions with retrieval
            inference_w_retrieval(script_args,data_path=known_path,save_path=known_path,data_type="known")

            #Construct DPO training data
            dpo_data = []
            with open(known_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            kg_data = [item for item in data if item['type'] == 'kg']
            kn_data = [item for item in data if item['type'] == 'kn']
            with open(unknown_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            ug_data = [item for item in data if item['type'] == 'ug']
            un_data = [item for item in data if item['type'] == 'un']

            #KG: known question with golden retrieval
            for item in kg_data:
                if item['answer'] and item['answer'] == item['ctx_answer']: #GT > IDK   
                    dpo_data.append({
                        "question": item['question'],
                        "ctxs": item['ctxs'],
                        'chosen': item['answer'],
                        'rejected': refusal_answer,
                        "type": "kg"
                    })
            #KN: known question with noisy retrieval
            for item in kn_data:
                prob = random.uniform(0,1)
                if prob >= 0.5: #GT > IDK
                    if item['answer']:
                        dpo_data.append({
                            "question": item['question'],
                            'ctxs': item['ctxs'],
                            'chosen': item['answer'],
                            'rejected': refusal_answer,
                            "type": "kn"
                        })
                else: #GT > WA1
                    if item['answer'] and item['ctx_wrong_answer']:
                        dpo_data.append({
                            "question": item['question'],
                            'ctxs': item['ctxs'],
                            'chosen': item['answer'],
                            'rejected': item['ctx_wrong_answer'],
                            "type": "kn"
                        })
            #UG: unknown question with golden retrieval
            for item in ug_data:
                prob = random.uniform(0,1)
                if prob >= 2/3: #GT > IDK
                    if item['ctx_answer']:
                        dpo_data.append({
                            "question": item['question'],
                            "ctxs": item['ctxs'],
                            'chosen': item['ctx_answer'],
                            'rejected': refusal_answer,
                            "type": "ug"
                        })
                elif prob >= 1/3: #GT > WA1
                    if item['ctx_answer'] and item['ctx_wrong_answer']:
                        dpo_data.append({
                            "question": item['question'],
                            "ctxs": item['ctxs'],
                            'chosen': item['ctx_answer'],
                            'rejected': item['ctx_wrong_answer'],
                            "type": "ug"
                        })
                else: #GT > WA2
                    if item['answer'] and item['ctx_answer']:
                        dpo_data.append({
                            "question": item['question'],
                            "ctxs": item['ctxs'],
                            'chosen': item['ctx_answer'],
                            'rejected': item['answer'],
                            "type": "ug"
                        })
            #UN: unknown question with noisy retrieval
            for item in un_data:
                prob = random.uniform(0,1)
                if prob >=2/3: #IDK > GT
                    dpo_data.append({
                        "question": item['question'],
                        "ctxs": item['ctxs'],
                        'chosen': refusal_answer,
                        'rejected': item['answers'][0],
                        "type": "un"
                    })
                elif prob >= 1/3: #IDK > WA1
                    if item['ctx_wrong_answer']:
                        dpo_data.append({
                            "question": item['question'],
                            "ctxs": item['ctxs'],
                            'chosen': refusal_answer,
                            'rejected': item['ctx_wrong_answer'],
                            "type": "un"
                        })
                else: #IDK > WA2
                    if item['answer']:
                        dpo_data.append({
                            "question": item['question'],
                            "ctxs": item['ctxs'],
                            'chosen': refusal_answer,
                            'rejected': item['answer'],
                            "type": "un"
                        })

            with open(os.path.join(data_path,'dpo_data.json'), 'w', encoding='utf-8') as f:
                json.dump(dpo_data, f, ensure_ascii=False, indent=2)
            print(f"DPO data generated for {source} has {len(dpo_data)} samples.")
            print(f"kg: {len([item for item in dpo_data if item['type'] == 'kg'])}")
            print(f"kn: {len([item for item in dpo_data if item['type'] == 'kn'])}")
            print(f"ug: {len([item for item in dpo_data if item['type'] == 'ug'])}")
            print(f"un: {len([item for item in dpo_data if item['type'] == 'un'])}")
        else:   
            #Generate four types of  data for evaluation(ug,un,kn,kg) (known forthreshold=1.0)
            data_path = os.path.join(script_args.data_path,model_name,source)
            unknown_path = os.path.join(data_path,'test','unknown.json')
            known_path = os.path.join(data_path,'test','known.json')
            eval_data = []
            #top-3
            with open(unknown_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                ctxs_pool = item['possible_golden_ctxs'] + item['possible_noisy_ctxs']
                if len(ctxs_pool) < 3:
                    continue
                sorted_ctxs_pool = sorted(ctxs_pool, key=lambda x: x['score'], reverse=True)
                ctxs_pool = sorted_ctxs_pool[:3]
                g_count = 0
                for ctx in ctxs_pool:
                    if ctx['type'] == 'g':
                        g_count += 1
                if g_count > 0:
                    #ug: unknown question with golden retrieval
                    eval_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "type": "ug"
                    })
                else:
                    #un: unknown question with noisy retrieval
                    eval_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "type": "un"
                    })

            with open(known_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                ctxs_pool = item['possible_golden_ctxs'] + item['possible_noisy_ctxs']
                if len(ctxs_pool) < 3:
                    continue
                sorted_ctxs_pool = sorted(ctxs_pool, key=lambda x: x['score'], reverse=True)
                ctxs_pool = sorted_ctxs_pool[:3]
                g_count = 0
                for ctx in ctxs_pool:
                    if ctx['type'] == 'g':
                        g_count += 1
                if g_count > 0:
                    #kg: known question with golden retrieval
                    eval_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "type": "kg"
                    })
                else:
                    #kn: known question with noisy retrieval
                    eval_data.append({
                        "question": item['question'],
                        "answers": item['answers'],
                        "ctxs": ctxs_pool,
                        "type": "kn"
                    })
            with open(os.path.join(data_path,'eval_data.json'), 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)
            print(f"Evaluation data generated for {source} has {len(eval_data)} samples.")




