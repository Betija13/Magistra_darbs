#!/bin/bash

chosen_prompt="Solve task below. Answer with (A),(B),(C),(D)"

python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE MAJOR_ANSWER
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE RERANK --REWARD_NAME_VALUE nvidia/nv-rerankqa-mistral-4b-v3
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE ANOTHER_LLM_RERANKING --REWARD_NAME_VALUE gemini-2.0-flash
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE ANOTHER_LLM_BEST_IDX --REWARD_NAME_VALUE gemini-2.0-flash
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE REWARD_MODEL --REWARD_NAME_VALUE internlm/internlm2-1_8b-reward
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE REWARD_MODEL --REWARD_NAME_VALUE Ray2333/GRM-Llama3-8B-rewardmodel-ft
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE REWARD_MODEL --REWARD_NAME_VALUE Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
python script_answers.py --TASK_VALUE DYNAMIC_REWARD_MM --PROMPTS_ITERATION "$chosen_prompt" \
--REWARD_METHOD_VALUE REWARD_MODEL --REWARD_NAME_VALUE internlm/internlm2-7b-reward
