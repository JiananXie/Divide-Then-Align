model_name_or_path="models/llama2"
datasets="nq,triviaqa,webq"
data_path="data_kbrag"
checkpoint_path="models/raat/best_checkpoint.bin"
infer_k=10
eval=False

CUDA_VISIBLE_DEVICES=3 python data_constructer.py \
    --model_name_or_path $model_name_or_path \
    --checkpoint_path $checkpoint_path \
    --datasets $datasets \
    --data_path $data_path \
    --infer_k $infer_k \
    --eval $eval