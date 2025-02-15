#!/bin/bash

# Models to test
MODELS=(
    "models/chatqa1.5"
)

# Data path
DATA_PATH="data_kbrag/llama3/evaluation_knowledge.json"  # Update this path
PREDICTION_PATH="result/3_chatqa1.5.json"
RESULT_PATH="results_baseline"

# Create results directory
mkdir -p $RESULT_PATH

# Run experiments for each model
for MODEL in "${MODELS[@]}"; do
    echo "Running experiments for model: $MODEL"
    
    # Logprob detection with different metrics and thresholds
    echo "Running logprob detection..."
    for METRIC in "min"; do
        for THRESHOLD in -2.0 -1.0 0.0; do
            echo "Logprob metric: $METRIC, threshold: $THRESHOLD"
            CUDA_VISIBLE_DEVICES=3 python baselines/logprob_detection.py \
                --model_name_or_path $MODEL \
                --data_path $DATA_PATH \
                --threshold $THRESHOLD \
                --metric $METRIC \
                --result_path $RESULT_PATH \
                --ctxs_num 3
        done
    done


    echo "Running p_true detection..."
    for THRESHOLD in 0.5 0.7 0.9; do
        echo "P-true threshold: $THRESHOLD"
        CUDA_VISIBLE_DEVICES=3 python baselines/p_true_detection.py \
            --model_name_or_path $MODEL \
            --data_path $PREDICTION_PATH \
            --threshold $THRESHOLD \
            --result_path $RESULT_PATH
    done
    
    # ICL detection with different numbers of examples
    echo "Running ICL detection..."
    NUM_EXAMPLES=3
    CUDA_VISIBLE_DEVICES=7 python baselines/icl_detection.py \
        --model_name_or_path $MODEL \
        --data_path $DATA_PATH \
        --num_examples $NUM_EXAMPLES \
        --result_path $RESULT_PATH \
        --ctxs_num 3            
    
    #Self-consistency detection
    echo "Running self-consistency detection..."
    infer_k=3  
    CUDA_VISIBLE_DEVICES=3 python baselines/self_consistency_detection.py \
        --model_name_or_path $MODEL \
        --data_path $DATA_PATH \
        --infer_k $infer_k \
        --result_path $RESULT_PATH \
        --ctxs_num 3

done

echo "All experiments completed. Results saved in $RESULT_PATH" 