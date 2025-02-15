import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def vllm_w_retrieval(args,data):
    sys = "You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences. If you don't know the answer and the following contexts do not contain the necessay information to answer the question, respond with 'This question is beyond the scope of my knowledge and the references, I don't know the answer'."
    sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'
    sys_ctxs = 'The following contexts will help you complete the question-and-answer pair.'
    rep_penalty = 1
    if 'llama2' in args.model_name_or_path or 'raat' in args.model_name_or_path:
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
        if not args.checkpoint_path:
            rep_penalty = 1.1
            examples = [
                '\nQuestion: What is the capital of France?\nAnswer: Paris.',
                '\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
                '\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
            ]
            prompt='{system_prompt}\n\n{ctx}\n{query}'
    elif 'llama3' in args.model_name_or_path:
        rep_penalty = 1
        examples = [
            '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
            '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
            '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
        ]
        prompt='{system_prompt}\n\n{ctx}\nDirectly answer the question without any other words.{query}'
    elif 'chatqa' in args.model_name_or_path:
        prompt='{system_prompt}\n\n{ctx}\n{query}'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True, load_format='auto', gpu_memory_utilization=0.95, max_num_batched_tokens=16384)
    if args.checkpoint_path:    
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

    sampling_param = SamplingParams(
                max_tokens=512,
                top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=rep_penalty,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        if 'chatqa' in args.model_name_or_path:
            if args.ctxs_num == 1:
                sentence = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.format(system_prompt=sys,ctx=sys_ctx+sample['ctxs'][0]['text'],query='Question:'+sample['question'])}],
                    tokenize=False,
                    add_generation_prompt=True,
            )
            else:
                ctxs = ''
                for i in range(args.ctxs_num):
                    ctxs += f'\nContext{i+1}:{sample["ctxs"][i]["text"]}'
                sentence = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.format(system_prompt=sys,ctx=sys_ctxs+ctxs,query='Question:'+sample['question'])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # print(sentence)
        elif ('llama2' in args.model_name_or_path or 'llama3' in args.model_name_or_path) and not args.checkpoint_path:
            if args.ctxs_num == 1:
                sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctx+sample['ctxs'][0]['text'],query='Question:'+sample['question'])
            else:
                ctxs = ''
                for i in range(args.ctxs_num):
                    ctxs += f'\nContext{i+1}:{sample["ctxs"][i]["text"]}'
                sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctxs+ctxs,query='Question:'+sample['question'])
        else:
            if args.ctxs_num == 1:
                sentence = prompt.format(system_prompt=sys,ctx=sys_ctx+sample['ctxs'][0]['text'],query='Question:'+sample['question'])
            else:
                ctxs = ''
                for i in range(args.ctxs_num):
                    ctxs += f'\nContext{i+1}:{sample["ctxs"][i]["text"]}'
                sentence = prompt.format(system_prompt=sys,ctx=sys_ctxs+ctxs,query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ret.append(generated_text)

    return ret