# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 0. imports
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Literal
import json
import torch
import math
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, load_dataset
import random
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import itertools
from trl import DPOConfig, DPOTrainer
from tuner.datacollator import PreferenceCollator
from tuner.dpotrainer import Trainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    checkpoint_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the checkpoint path"},
    )
    data_dir: Optional[str] = field(
        default="data/dpo/train",
        metadata={"help": "the location of the data path"},
    )
    cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the cache path of training data"},
    )
    data_size: Optional[int] = field(
        default=10000,
        metadata={"help": "the number of data to use"},
    )
    idk_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "the ratio of un data"}
    )
    aux_loss: Optional[str] = field(
        default="",
        metadata={"help": "the aux loss type",
                  "choices": ["none", "sft", "cls","mix"]}
    )
    coe_cls: Optional[float] = field(
        default=0.5,
        metadata={"help": "the coefficient of cls loss"}
    )
    quadrant_num: Optional[int] = field(
        default=4,
        metadata={"help": "the number of quarants"}
    )
    coe_sft: Optional[float] = field(
        default=0.5,
        metadata={"help": "the coefficient of sft loss"}
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "the ratio of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=16, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=3072, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=500, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def prompt_template(script_args,tokenizer,ctxs,query):
    sys = "You need to complete the question-and-answer pair. The answers should be short phrases or entities, not full sentences. If you don't know the answer and the following contexts do not contain the necessary information to answer the question, respond with 'This question is beyond the scope of my knowledge and the references, I don't know the answer'."
    # sys_ctx = 'The following context will help you complete the question-and-answer pair.\nContext:'
    sys_ctxs = 'The following contexts will help you complete the question-and-answer pair.'
    if script_args.model_name_or_path.endswith('llama2'):
        prompt='<<SYS>>\n{system_prompt}\n<</SYS>>\n\n[INST] {ctx}\n{query} [/INST]'
        content = ''
        for i in range(3):
            content += f'\nContext{i+1}:{ctxs[i]["text"]}'
        return prompt.format(system_prompt=sys, ctx=sys_ctxs+content, query=query)
    elif script_args.model_name_or_path.endswith('chatqa1.5'):
        content = ''
        template = '{system_prompt}\n\n{ctx}\n{query}'
        for i in range(3):
            content += f'\nContext{i+1}:{ctxs[i]["text"]}'
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": template.format(system_prompt=sys,ctx=sys_ctxs+content,query=query)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    else:
        raise ValueError(f"Unsupported model: {script_args.model_name_or_path}")
    
def data_sample(data_list,sample_size):
    if len(data_list) >= sample_size:
        return random.sample(data_list, sample_size)
    else:
        return data_list + random.choices(data_list,k=sample_size-len(data_list))

def map_type(type):
    if type == "kg":
        return 0
    elif type == "kn":
        return 1
    elif type == "ug":
        return 2
    elif type == "un":
        return 3
    
def dpo_dataset(script_args,tokenizer):
    dataset = {"prompt": [], "chosen": [], "rejected": [], "label": []}
    data_size = script_args.data_size
    idk_ratio = script_args.idk_ratio

    # 检查是否存在cache文件
    if not script_args.cache_path:
        raise ValueError("cache_path is required to load cached dataset or save training data")
    
    if os.path.exists(script_args.cache_path):
        print(f"Loading dataset from cache: {script_args.cache_path}")
        with open(script_args.cache_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    else:
        chosen_rejected_data = []
        data_subdir = [source for source in os.listdir(script_args.data_dir) if os.path.isdir(os.path.join(script_args.data_dir, source))]
        for subdir in data_subdir:
            filename = os.path.join(script_args.data_dir, subdir, 'dpo_data.json')
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chosen_rejected_data.append(data)

        chosen_rejected_data = list(itertools.chain(*chosen_rejected_data))

        all_data = []

        kg_data = [item for item in chosen_rejected_data if item['type'] == 'kg']
        kn_data = [item for item in chosen_rejected_data if item['type'] == 'kn']
        ug_data = [item for item in chosen_rejected_data if item['type'] == 'ug']
        un_data = [item for item in chosen_rejected_data if item['type'] == 'un']

        answerable_size = math.floor(data_size * (1-idk_ratio) // 3)
        abstain_size = data_size - answerable_size * 3
        all_data.append(data_sample(kg_data,answerable_size))
        all_data.append(data_sample(kn_data,answerable_size))
        all_data.append(data_sample(ug_data,answerable_size))
        all_data.append(data_sample(un_data,abstain_size))
        all_data = list(itertools.chain(*all_data))
        print(f'''Total training data (train:eval=9:1): {len(all_data)},
                    kg_data:{answerable_size},
                    kn_data:{answerable_size},
                    ug_data:{answerable_size},
                    un_data:{abstain_size}''')
        
        random.shuffle(all_data)
        
        for item in all_data:
            dataset["prompt"].append(prompt_template(script_args,tokenizer,ctxs=item['ctxs'],query=f"Question: {item['question']}"))
            if script_args.model_name_or_path.endswith('chatqa1.5'):
                dataset["chosen"].append(item['chosen'])
                dataset["rejected"].append(item['rejected'])
            elif script_args.model_name_or_path.endswith('llama2'):
                dataset["chosen"].append('\nAnswer: '+item['chosen'])
                dataset["rejected"].append('\nAnswer: '+item['rejected'])
            dataset["label"].append(map_type(item['type']))
        print(f"Saving dataset to cache: {script_args.cache_path}")
        with open(script_args.cache_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    dataset = Dataset.from_dict(dataset)
    
    return dataset

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    
    if script_args.checkpoint_path != "":
        print("Loading the checkpoint...")
        state_dict = torch.load(script_args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Checkpoint is loaded.")

    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    total_dataset = dpo_dataset(script_args,tokenizer)
    train_dataset, eval_dataset = total_dataset.train_test_split(test_size=0.1).values()

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length,
    )
    # print(train_dataset[0],eval_dataset[0])
    # print(max(len(x["prompt"]) + len(x["chosen"]) for x in train_dataset))
    # print(max(len(x["prompt"]) + len(x["rejected"]) for x in train_dataset))
    # print(max(len(x["prompt"]) + len(x["chosen"]) for x in eval_dataset))
    # print(max(len(x["prompt"]) + len(x["rejected"]) for x in eval_dataset))

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. initialize training arguments:
    training_args = DPOConfig(
        save_only_model=True,
        generate_during_eval=True,
        num_train_epochs=script_args.num_train_epochs,
        save_total_limit=1,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        load_best_model_at_end=True
    )

    # 5. initialize the DPO trainer
    dpo_trainer = Trainer(
        aux_loss=script_args.aux_loss,
        coe_cls=script_args.coe_cls,
        coe_sft=script_args.coe_sft,
        quadrant_num=script_args.quadrant_num,
        data_collator=PreferenceCollator(pad_token_id=tokenizer.pad_token_id),
        model=model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()

    dpo_trainer.model.config.use_cache = True
    dpo_trainer.save_state()

    # 7. save final checkpoint.bin
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    os.makedirs(output_dir,exist_ok=True)
    merged_model = dpo_trainer.model.merge_and_unload()
    # dpo_trainer.model.save_pretrained(output_dir,max_shard_size="16GB",safe_serialization=False)
    merged_model.save_pretrained(output_dir,max_shard_size="16GB",safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

  