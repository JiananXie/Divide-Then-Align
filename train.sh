export NCCL_P2P_LEVEL="NVL"
export OMP_NUM_THREADS=8

model_name_or_path="models/llama2"
checkpoint_path="models/raat/best_checkpoint.bin"
data_dir="data_kbrag/llama2"
data_size=5000
learning_rate=5e-5
per_device_train_batch_size=16
gradient_accumulation_steps=2
idk_ratio=0.7
output_dir="./checkpoints/raat_ir0.7_d5k_0.5mix1.0"
cache_path="data_kbrag/llama2/training_data_ir0.7_d5k.json"
save_steps=10
eval_steps=10
lora_alpha=64
lora_r=64
epochs=3
aux_loss="mix"
coe_cls=0.5
coe_sft=1.0

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=5 accelerate launch \
    --num_processes 1 \
    --main_process_port 29505 \
    dpo_trainer.py \
    --beta 0.1 \
    --model_name_or_path $model_name_or_path \
    --checkpoint_path $checkpoint_path \
    --learning_rate $learning_rate \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --data_dir $data_dir \
    --cache_path $cache_path \
    --data_size $data_size\
    --idk_ratio $idk_ratio \
    --load_in_4bit False\
    --output_dir $output_dir\
    --save_steps $save_steps\
    --eval_steps $eval_steps\
    --num_train_epochs $epochs\
    --lora_alpha $lora_alpha\
    --lora_r $lora_r\
    --aux_loss $aux_loss\
    --coe_cls $coe_cls\
    --coe_sft $coe_sft\
    --report_to none