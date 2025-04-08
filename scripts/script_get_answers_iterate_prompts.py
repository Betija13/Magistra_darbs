import csv
import os
from loguru import logger
from typing import List
from controllers.AiLLM import ControllerAiLLM
from controllers.RewardMethods import RewardMethods
from controllers.AnswerMethods import AnswerMethods
from utils.file_utils import FileUtils
from utils.result_utils import ResultUtils
from models.DataClass.DataResults import DataResults
from models.DataClass.InfoResults import InfoResults
from models.Enums.AnswerType import AnswerType
from models.Enums.Method import Method
from models.Enums.RewardMethod import RewardMethod
from models.Enums.Datasets import Datasets
from models.constants import human_prompts, mutated_task_prompts_AQuA_RAT, system_prompts_output, system_prompts_static
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict

controller_answers = AnswerMethods()
CUSTOM_NAME = 'mutated_prompts_combined'
TOTAL_COUNT = 100
TEMPERATURE = 0.0
ANSWER_COUNT = 1
METHOD = Method.A_1.value
METHOD_NAME_FILE = 'a_1'
REWARD_METHOD = None #RewardMethod.MAJOR.value
MODEL_NAME = 'gpt-4o' #'o3-mini', 'gpt-4o-mini', 'gpt-4o'
PREDIFINED_DATASETS: List[str] = [Datasets.AQUA.value]
PREDIFINED_FILES: List[str] = ['data_normalized_test.csv']


folder = '../datasets'
filename = 'info_results.csv'
file_path_info_results = os.path.join(folder, filename)
previous_highest_id = FileUtils.get_highest_id_from_csv(file_path_info_results)

fieldnames = [field.name for field in DataResults.__dataclass_fields__.values()]
current_date = datetime.now().strftime('%d-%m-%Y')
all_dataset_folders = FileUtils.get_all_files_and_folders('../datasets', only_folders=True)
all_dataset_folders = FileUtils.select_folders_datasets(all_dataset_folders, predifined_names=PREDIFINED_DATASETS)


existing_file_paths_results = set()
with open(file_path_info_results, 'r', encoding='utf-8') as resultsfile:
    reader = csv.DictReader(resultsfile)
    for row in reader:
        existing_file_paths_results.add(row['result_file_name'])

custom_name_str = f'_{CUSTOM_NAME}' if CUSTOM_NAME else ''
for prompt_idx, system_prompt_task in enumerate(mutated_task_prompts_AQuA_RAT):
    system_prompt = f"{system_prompt_task}\n\n{system_prompts_output[AnswerType.MULTIPLE_CHOICE.value]}\n\n{system_prompts_static[AnswerType.MULTIPLE_CHOICE.value]}"
    for folder in all_dataset_folders:
        # question_file = f'../datasets/{folder}/data/data_normalized.csv'
        # if not os.path.exists(question_file):
        path_to_question_file = f'../datasets/{folder}/data'
        files_in_folder = FileUtils.get_all_files_and_folders(path_to_question_file, only_folders=False)
        if len(files_in_folder) > 0:
            if len(files_in_folder) > 1:
                question_files = FileUtils.select_folders_datasets(files_in_folder, predifined_names=PREDIFINED_FILES)
            else:
                question_files = files_in_folder
            question_files_it = [os.path.join(path_to_question_file, q_file) for q_file in question_files]
        else:
            logger.error(f'No files in {path_to_question_file} folder')
            continue
        for question_file in question_files_it:
            print(question_file)
            question_filename = question_file.split('/')[-1].split('\\')[-1].split('.')[0]

            result_file = f'../datasets/{folder}/results/{METHOD_NAME_FILE}_{current_date}_{question_filename}{custom_name_str}_{prompt_idx}.csv'
            # Open the results file in append mode and read existing ids
            existing_ids = set()
            system_prompt_info = None
            human_prompt_info = None

            try:
                with open(result_file, 'r', encoding='utf-8') as resultsfile:
                    reader = csv.DictReader(resultsfile)
                    for row in reader:
                        existing_ids.add(int(row['id']))
            except FileNotFoundError:
                with open(result_file, 'w', newline='', encoding='utf-8') as resultsfile:
                    writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                    writer.writeheader()

            # Get unique answer types
            unique_answer_types = set()
            with open(question_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    unique_answer_types.add(row['answer_type'])


            with open(question_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                with open(result_file, 'a', newline='', encoding='utf-8') as resultsfile:
                    writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                    for idx, row in tqdm(enumerate(reader), total=TOTAL_COUNT):
                        if idx >= TOTAL_COUNT:
                            break
                        answer_type = row['answer_type']
                        # system_prompt = system_prompts[answer_type]
                        choices_str_info = "Choices:\n```\n{choices}\n```\n" if row['choices'] else ''
                        facts_str_info = "Facts:\n```\n{facts}\n```\n" if row['facts'] else ''
                        start_human_prompt_info = 'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else 'Problem:\n```\n{question}\n```\n'
                        end_human_prompt = human_prompts[answer_type]

                        this_human_prompt_info = f"{start_human_prompt_info}{choices_str_info}{facts_str_info}{end_human_prompt}"
                        if len(unique_answer_types) > 1:
                            if human_prompt_info is None:
                                human_prompt_info = f"[{answer_type}]\n{this_human_prompt_info}"
                            else:
                                if this_human_prompt_info not in human_prompt_info:
                                    human_prompt_info += f"\n------\n[{answer_type}]\n{this_human_prompt_info}"
                            if system_prompt_info is None:
                                system_prompt_info = f"[{answer_type}]\n{system_prompt}"
                            else:
                                if system_prompt not in system_prompt_info:
                                    system_prompt_info += f"\n------\n[{answer_type}]\n{system_prompt}"
                        elif len(unique_answer_types) == 1:
                            human_prompt_info = this_human_prompt_info
                            system_prompt_info = system_prompt

                        if int(row['id']) not in existing_ids:
                            question = row['question']
                            qid = row['quid']
                            answer = row['answer']
                            answer_word = row['answer_word'] if row['answer_word'] != '' else None
                            choices_str = f"Choices:\n```\n{row['choices']}\n```\n" if row['choices'] else ''
                            if answer_word is None and answer_type == AnswerType.MULTIPLE_CHOICE.value:
                                answer_choice = [ans for ans in choices_str.split('\n') if ans.startswith(answer)]
                                if len(answer_choice) == 1:
                                    answer_word = answer_choice[0].split(f'{answer})')[-1].strip()
                            if row['choices']:
                                question_and_choices = f"{question}\n{row['choices']}"
                            else:
                                question_and_choices = question
                            facts_str = f"Facts:\n```\n{row['facts']}\n```\n" if row['facts'] else ''
                            start_human_prompt = f'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else f'Problem:\n```\n{question}\n```\n'
                            human_prompt = f"{start_human_prompt}{choices_str}{facts_str}{end_human_prompt}"
                            if METHOD == Method.A_2.value and REWARD_METHOD == RewardMethod.MAJOR.value:
                                answer_results = controller_answers.get_n_sampling_llm_answer_majority(
                                    system_prompt=system_prompt, human_prompt=human_prompt, response_count=ANSWER_COUNT,
                                    temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                                    ground_truth_answer=answer, ground_truth_answer_word=answer_word
                                )
                            elif METHOD == Method.A_1.value and REWARD_METHOD is None:
                                answer_results = controller_answers.get_zero_shot_answer(
                                    system_prompt=system_prompt, human_prompt=human_prompt,
                                    temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                                    ground_truth_answer=answer, ground_truth_answer_word=answer_word
                                )
                                ANSWER_COUNT = 1

                            data_results = DataResults(
                                id=int(row['id']),
                                quid=qid,
                                question=question_and_choices,
                                true_answer=answer,
                                llm_answer=answer_results.llm_answer_unedited,
                                correct=answer_results.correct,
                                llm_answer_chosen=answer_results.chosen_answer,
                                reward_method=REWARD_METHOD
                            )
                            writer.writerow(asdict(data_results))
            previous_highest_id += 1
            info_result = InfoResults(
                id=previous_highest_id,
                date=current_date,
                dataset_name=folder,
                method=METHOD,
                finished=True,
                system_prompt=system_prompt_info,
                human_prompt=human_prompt_info,
                temperature=TEMPERATURE,
                response_count=ANSWER_COUNT,
                reward_method=REWARD_METHOD,
                llm_model=controller_answers.controller_ai.model.name if MODEL_NAME is None else MODEL_NAME
            )
            info_result.result_file_name = resultsfile.name
            info_result.count = TOTAL_COUNT
            numeric_results = ResultUtils.count_correct_values(resultsfile.name)
            info_result.accuracy = numeric_results.accuracy_score
            info_result.percentage_of_short_answers = numeric_results.percentage_of_short_answers

            # Convert the dataclass instance to a dictionary
            data_to_append = asdict(info_result)

            # Append the data to the CSV file
            if info_result.result_file_name not in existing_file_paths_results:
                with open(file_path_info_results, 'a', newline='', encoding='utf-8') as csvfile:
                    writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
                    writer_info.writerow(data_to_append)
            else:  # TODO edit the existing row
                previous_highest_id -= 1
