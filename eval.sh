model_name_or_path="checkpoints/raat_ir0.7_d5k_0.5mix1.0/final_checkpoint"
benchmark="knowledge"
data_path="data_kbrag"
datasets="nq,triviaqa,webq"
ctxs_num=3
total_size=3000
cache_path="data_kbrag/llama2/evaluation_knowledge.json"
result_path="result"
seed=0

CUDA_VISIBLE_DEVICES=1 python evaluations.py \
    --model_name_or_path $model \
    --benchmark $benchmark \
    --data_path $data_path \
    --datasets $datasets \
    --cache_path $cache_path \
    --ctxs_num $ctxs_num \
    --total_size $total_size \
    --seed $seed \
    --result_path $result_path