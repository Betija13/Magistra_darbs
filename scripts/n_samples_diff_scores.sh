#!/bin/bash


# Define the array with your model names
models=(
"internlm/internlm2-1_8b-reward"
"Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
"Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
"RLHFlow/ArmoRM-Llama3-8B-v0.1"
"nicolinho/QRM-Llama3.1-8B-v2"
"sfairXC/FsfairX-LLaMA3-RM-v0.1"
"maywell/Better-PairRM"
"llm-blender/PairRM"
"OpenAssistant/reward-model-deberta-v3-large-v2"
"Ray2333/GRM-gemma2-2B-rewardmodel-ft"
"Ray2333/GRM-Llama3-8B-rewardmodel-ft"
"internlm/internlm2-7b-reward"
)

original_files=(
"../datasets/MMLU/results/A_N_SAMPLING_MAJOR_ANSWER__O-STRUCTURED_COT___I-N_05-05-2025_F-data_normalized_STEM_dev_3.csv"
"../datasets/MMLU/results/A_N_SAMPLING_MAJOR_ANSWER__O-STRUCTURED_COT___I-N_05-05-2025_F-data_normalized_STEM_dev_4.csv"
"../datasets/MMLU/results/A_N_SAMPLING_MAJOR_ANSWER__O-STRUCTURED_COT___I-N_05-05-2025_F-data_normalized_STEM_dev_1.csv"
"../datasets/MMLU/results/A_N_SAMPLING_MAJOR_ANSWER__O-STRUCTURED_COT___I-N_05-05-2025_F-data_normalized_STEM_dev_2.csv"
)

for original_file in "${original_files[@]}"; do
    echo "Processing file: $original_file"
    python script_answers.py --TASK_VALUE N_SAMPLE_DIFF_SCORE_OTHER --ORIGINAL_FILE "$original_file"

    for val in "${models[@]}"; do
        echo "  Running for model: $val"
        python script_answers.py --TASK_VALUE N_SAMPLE_DIFF_SCORE_RM --REWARD_NAME_VALUE "$val" --ORIGINAL_FILE "$original_file"
    done
done