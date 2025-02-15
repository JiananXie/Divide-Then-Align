import torch
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def extract_answer(args,text):
    if 'llama2' in args.model_name_or_path or 'raat' in args.model_name_or_path:
        pattern =  r"Answer:(.*)"
        match = re.search(pattern, text)
        if match:
            extracted_text = match.group(1)
            if extracted_text!='' and extracted_text[-1] == '.':
                extracted_text = extracted_text[:-1]
            return extracted_text
        else:
            return False
    elif 'llama3' in args.model_name_or_path:
        if text:
            if '.Question' in text:
                text = text.split('.Question')[0].strip()
            elif 'Question' in text:
                text = text.split('Question')[0].strip()
            elif 'Context' in text:
                text = text.split('Context')[0].strip()
            elif '.Context' in text:
                text = text.split('.Context')[0].strip()
            elif text[-1] == '.' or text[-1] == '\n':
                text = text[:-1]
            text_comma = text.split(',')
            if len(text_comma) > 2:
                text = text_comma[0].strip()
        else:
            return False
        pattern = r"Answer:(.*?)(?=\n|$)"
        match = re.search(pattern, text)
        if match:
            extracted_text = match.group(1)
            if extracted_text!='' and extracted_text[-1] == '.':
                extracted_text = extracted_text[:-1]
            return extracted_text
        else:
            return text
    elif 'chatqa' in args.model_name_or_path:
        if text:
            if text[-1] == '.' or text[-1] == '\n':
                text = text[:-1]
            text_comma = text.split(',')
            if len(text_comma) > 2:
                text = text_comma[0].strip()
            return text
        else:
            return False



def vllm_wo_retrieval(args,data):
    rep_penalty = 1
    if 'llama2' in args.model_name_or_path:
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        if args.checkpoint_path:
            examples = [
                '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
                '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
                '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
            ]
            prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {query} [/INST]'
        else:
            examples = [
                '\nQuestion: What is the capital of France?\nAnswer: Paris.',
                '\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
                '\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
            ]
            rep_penalty = 1.2
            prompt='{system_prompt}\n\n{query}'
    elif 'llama3' in args.model_name_or_path:
        rep_penalty = 1.1
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        examples = [
            '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
            '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
            '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
        ]
        prompt='{system_prompt}\n\nDirectly answer the question without any other words.{query}'
    elif 'chatqa' in args.model_name_or_path:
        sys = 'You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences.'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        prompt = '{system_prompt}\n\n{query}'

    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True, load_format='auto', gpu_memory_utilization=0.95, max_num_batched_tokens=16384)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

    sampling_param = SamplingParams(
                n=args.infer_k,
                max_tokens=512,
                top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=rep_penalty,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        if 'chatqa' in args.model_name_or_path:
            sentence = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.format(system_prompt=sys,query='Question:'+sample['question'])}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_k_text = []
        for i in range(args.infer_k):
            generated_text = output.outputs[i].text
            generated_k_text.append(generated_text)
        ret.append(generated_k_text)

    return ret

def vllm_w_retrieval(args,data):
    # sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'
    sys_ctxs = 'The following contexts will help you complete the question-and-answer pair.'
    rep_penalty = 1
    if 'llama2' in args.model_name_or_path:
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        if args.checkpoint_path:
            examples = [
                '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
                '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
                '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
            ]
            prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
        else:
            examples = [
                '\nQuestion: What is the capital of France?\nAnswer: Paris.',
                '\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
                '\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
            ]
            rep_penalty = 1.1
            prompt='{system_prompt}\n\n{ctx}\n{query}'
    elif 'llama3' in args.model_name_or_path:
        rep_penalty = 1.1
        sys = 'You need to complete the question-and-answer pair following the format provided in the example. The answers should be short phrases or entities, not full sentences. Here are some examples to guide you.'
        examples = [
            '\nExample 1:\nQuestion: What is the capital of France?\nAnswer: Paris.',
            '\nExample 2:\nQuestion: Who invented the telephone?\nAnswer: Alexander Graham Bell.',
            '\nExample 3:\nQuestion: Which element has the atomic number 1?\nAnswer: Hydrogen.'
        ]
        prompt='{system_prompt}\n\n{ctx}\nDirectly answer the question without any other words.{query}'
    elif 'chatqa' in args.model_name_or_path:
        sys = 'You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences.'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        prompt = '{system_prompt}\n\n{ctx}\n{query}'
    
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1,  tokenizer_mode='auto',
                trust_remote_code=True, load_format='auto', gpu_memory_utilization=0.95, max_num_batched_tokens=16384)
    if args.checkpoint_path:    
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

    sampling_param = SamplingParams(
                n=args.infer_k,
                max_tokens=512,
                top_k=50,
                top_p=0.6,
                temperature=0.3,
                repetition_penalty=rep_penalty,
                stop_token_ids=[7])
    sens = []
    for sample in data:
        if 'chatqa' in args.model_name_or_path:
            ctxs = ''
            for i in range(3):
                ctxs += f'\nContext{i+1}:{sample["ctxs"][i]["text"]}'           
            sentence = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.format(system_prompt=sys,ctx=sys_ctxs+ctxs,query='Question:'+sample['question'])}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            ctxs = ''
            for i in range(3):
                ctxs += f'\nContext{i+1}:{sample["ctxs"][i]["text"]}'
            sentence = prompt.format(system_prompt=sys+examples[0]+examples[1]+examples[2],ctx=sys_ctxs+ctxs,query='Question:'+sample['question'])
        sens.append(sentence)

    outputs = llm.generate(sens, sampling_params=sampling_param)
    ret = []
    for output in outputs:
        prompt = output.prompt
        generated_k_text = []
        for i in range(args.infer_k):
            generated_text = output.outputs[i].text
            generated_k_text.append(generated_text)
        ret.append(generated_k_text)

    return ret