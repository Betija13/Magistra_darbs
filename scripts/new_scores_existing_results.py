import csv
from loguru import logger
from dataclasses import asdict
from tqdm import tqdm
from datetime import datetime
from datetime import datetime
from controllers.RewardMethods import RewardMethods
from models.Enums.RewardMethod import RewardMethod
from models.DataClass.DataResults import DataResults
from models.DataClass.InfoResults import InfoResults
from utils.file_utils import FileUtils
from utils.result_utils import ResultUtils
reward_methods = RewardMethods()

custom_name = '_REWARD_LLM_GEMINI_Scores'
date = datetime.now().strftime('%d-%m-%Y')
used_reward_method = RewardMethod.LLM_O_B_I.value  # TODO change
reward_name = 'gemini-2.0-flash'  # TODO change

file_path = '../datasets/AQuA-RAT/results/A_N_SAMPLING_17-04-2025_data_normalized_test.csv'

question_filename = 'data_normalized_test'
reward_str = f"_{used_reward_method}"# if REWARD_METHOD else ''
until_satisfied_mut_str = ''
input_format = '_I-S_'
folder = 'AQuA-RAT'
result_file = (f'../datasets/{folder}/results/A_N_SAMPLING_{until_satisfied_mut_str}{reward_str}__'
               f'O-STRUCTURED_COT__{input_format}_{date}_F-{question_filename}'
               f'{custom_name}.csv')
file_path_new = f'../datasets/AQuA-RAT/results/A_N_SAMPLING_{date}_data_normalized_test_{custom_name}_TD.csv'
info_results_path = '../datasets/info_results.csv'

info_results_object = None
with open(info_results_path, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['result_file_name'] == file_path:
            if info_results_object is not None:
                logger.error("Info row is already something")
            info_results_object = InfoResults(**row)
a = 0
info_results_object.date = date

info_results_object.reward_method = used_reward_method
info_results_object.reward_name = reward_name
info_results_object.result_file_name = file_path_new
existing_ids = set()
fieldnames = [field.name for field in DataResults.__dataclass_fields__.values()]
try:
    with open(file_path_new, 'r', encoding='utf-8') as resultsfile:
        logger.warning(f"File {file_path_new} already exists")
        reader = csv.DictReader(resultsfile)
        for row in reader:
            existing_ids.add(int(row['id']))
except FileNotFoundError:
    with open(file_path_new, 'w', newline='', encoding='utf-8') as resultsfile:
        writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
        writer.writeheader()
# Read and update rows
updated_rows = []
with open(file_path_new, 'a', newline='', encoding='utf-8') as resultsfile:
    writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        for row in tqdm(reader, total=info_results_object.count):
            if int(row['id']) in existing_ids:
                continue
            # Access data using header names
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
            # question_only = question.split('\n')[0]
            # choices = '\n'.join(question.split('\n')[1:])
            # question_llm = f"Problem:\n```\n{question_only}\n```\nChoices:\n```\n{choices}\n```\n"
            # answer, score = reward_methods.get_reranking_model_best_answer(question, cot_parts)
            # final_letter_answer = reward_methods.majority_element(final_answer_parts)

            # answer, score = reward_methods.get_reward_model_internlm_best_answer(question=question, answer_options=cot_parts)
            # chosen_idx = cot_parts.index(answer)

            # TODO reranker as LLM
            if used_reward_method == RewardMethod.LLM_O_R.value:
                answer_obj = reward_methods.get_llm_best_answer_reranker(question=question, answer_options=cot_parts)
                score = answer_obj.answer_score
                chosen_answer = answer_obj.chosen_answer
                chosen_idx = cot_parts.index(chosen_answer)
            elif used_reward_method == RewardMethod.LLM_O_B_I.value:
                answer_obj = reward_methods.get_llm_best_answer_best_idx(question=question, answer_options=cot_parts)
                score = answer_obj.answer_score
                chosen_answer = answer_obj.chosen_answer
                chosen_idx = cot_parts.index(chosen_answer)
            # # system_prompt = "Provide the index of the best answer based on your analysis. Use logical reasoning " \
            # #                 "and contextual understanding to determine the most appropriate answer. "
            # system_prompt = "Provide scores for each of answer options to the question based on your analysis. Use logical reasoning " \
            #                 "and contextual understanding to determine the most appropriate answer for question and give that the highest score (10). "
            # answers_str = '\n\n'.join([f"ANSWER {idx}:\n{answer}" for idx, answer in enumerate(cot_parts)])
            # human_prompt = f'Question:\n```\n{question}\n```\nAnswer options:\n```\n{answers_str}\n```'
            # # BestIdx
            # # llm_answer_ranking = reward_methods.ai_llm.prompt_gemini_ranking(system_prompt=system_prompt, human_prompt=human_prompt)
            # # chosen_idx = llm_answer_ranking.best_idx
            # # score = llm_answer_ranking.best_answer_score
            # #Scores
            # llm_answer_ranking = reward_methods.ai_llm.prompt_gemini_ranking(system_prompt=system_prompt,
            #                                                                  human_prompt=human_prompt)
            # try:
            #     rankings = llm_answer_ranking.answer_rankings
            #     rankings_sorted = sorted(rankings, key=lambda r: r.answer_score, reverse=True)
            #     best_answer = rankings_sorted[0]
            #     chosen_idx = best_answer.answer_idx
            #     score = best_answer.answer_score
            # except Exception as e:
            #     a=0

            # if final_letter_answer is not None:
            #     chosen_idx = final_answer_parts.index(final_letter_answer)
            #
            final_str = answers_list[chosen_idx]
            final_letter_answer = final_answer_parts[chosen_idx]
            # chosen_answer = row['llm_answer_chosen']
            true_answer = row['true_answer']
            correct_answer = False
            if final_letter_answer == true_answer:
                correct_answer = True
            row['llm_answer_chosen'] = final_str
            # row['llm_answer_chosen'] = final_letter_answer
            row['reward_score'] = score
            row['reward_method'] = used_reward_method
            row['correct'] = correct_answer
            updated_rows.append(row)
            writer.writerow(row)
# with open(file_path_new, 'w', encoding='utf-8', newline='') as csv_file:
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(updated_rows)
current_result_id = FileUtils.get_highest_id_from_csv(info_results_path) + 1
info_results_object.id = current_result_id
numeric_results = ResultUtils.count_correct_values(file_path_new)
info_results_object.accuracy = numeric_results.accuracy_score
info_results_object.percentage_of_short_answers = numeric_results.percentage_of_short_answers

data_to_append = asdict(info_results_object)

with open(info_results_path, 'a', newline='', encoding='utf-8') as csvfile:
    writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
    writer_info.writerow(data_to_append)
