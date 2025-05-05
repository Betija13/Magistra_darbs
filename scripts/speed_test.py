import argparse
import sys
import os
import shutil
import time
import csv
from loguru import logger
from dotenv import dotenv_values
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from models.Enums.RewardModelNames import RewardModelNames
from controllers.RewardMethods import RewardMethods

env_path = '.env'
if os.path.exists("../.env"):
    env_path = "../.env"
elif os.path.exists(".env"):
    env_path = ".env"
elif os.path.exists("../../.env"):
    env_path = "../../.env"
else:
    logger.critical("No .env file found")
config = dotenv_values(env_path)

PATH_HF_MODELS = config.get('PATH_HF_MODELS')

list_rewards = [rn for rn in RewardModelNames]


parser = argparse.ArgumentParser()
parser.add_argument('--r_idx', type=int, required=True)
args = parser.parse_args()
reward_name = list_rewards[args.r_idx] # 12 for testing
reward_methods = RewardMethods(reward_name=reward_name)

file_path = '../datasets/AQuA-RAT/results/A_N_SAMPLING_17-04-2025_data_normalized_test.csv'

with open(file_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames
    start_time = time.time()
    for idx, row in enumerate(reader):
        if idx >= 10:
            break
        llm_answer = row['llm_answer']
        answers_list = llm_answer.split('\n------\n')
        cot_parts = []
        final_answer_parts = []
        for ans_n in answers_list:
            cot, answer_f = ans_n.split('ANSWER_AS_LETTER:')
            cot = cot.replace('SOLUTION_EXPLANATION:', '').strip()
            answer_f = answer_f.strip()
            cot_parts.append(cot)
            final_answer_parts.append(answer_f)
        question = row['question']
        answer_obj = reward_methods.get_reward_model_best_answer(
            question=question, answer_options=cot_parts
        )
        if answer_obj is None:
            print('Something went wrong: answer_object is None...')

end_time = time.time()

elapsed_time = end_time - start_time
logger.info(f"Time for {reward_name.value}: {elapsed_time} seconds")
logger.success(f"Average time for 1 iteration {reward_name.value} : {elapsed_time/10} seconds")

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path_n = os.path.join(dirpath, filename)
            if not os.path.islink(file_path_n):  # Skip symbolic links
                total_size += os.path.getsize(file_path_n)
    return total_size

reward_name_str = reward_name.value
username, modelname = reward_name_str.split('/')
model_str = f'models--{username}--{modelname}'
path_model = f"{PATH_HF_MODELS}/{model_str}"
if os.path.exists(path_model):
    size_model = get_folder_size(path_model)
    size_gb = size_model / (1024 ** 3)
    logger.debug(f"Size of model {reward_name_str}: {size_gb:.2f} GB")
    shutil.rmtree(path_model)
else:
    logger.error(f"Path {path_model} does not exist")
