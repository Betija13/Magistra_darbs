import csv
import os
from controllers.AiLLM import ControllerAiLLM
from controllers.RewardMethods import RewardMethods
from utils.file_utils import FileUtils
from utils.result_utils import ResultUtils
from models.DataClass.DataResults import DataResults
from models.DataClass.InfoResults import InfoResults
from models.Enums.AnswerType import AnswerType
from models.Enums.Method import Method
from models.Enums.RewardMethod import RewardMethod
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
controller_ai = ControllerAiLLM()

CUSTOM_NAME = ''
TOTAL_COUNT = 100
TEMPERATURE = 1
ANSWER_COUNT = 3
METHOD = Method.A_2.value
METHOD_NAME_FILE = 'a_2'
REWARD_METHOD = RewardMethod.MAJOR.value
MODEL_NAME = None#'o3-mini'

folder = '../datasets'
filename = 'info_results.csv'
file_path_info_results = os.path.join(folder, filename)
previous_highest_id = FileUtils.get_highest_id_from_csv(file_path_info_results)

fieldnames = [field.name for field in DataResults.__dataclass_fields__.values()]
current_date = datetime.now().strftime('%d-%m-%Y')
all_dataset_folders = FileUtils.get_all_files_and_folders('../datasets', only_folders=True)
all_dataset_folders = FileUtils.select_folders_datasets(all_dataset_folders)

system_prompts = {
    AnswerType.BOOL.value: "Work out an answer to the commonsense reasoning question above, and then answer yes or no.",
    AnswerType.MULTIPLE_CHOICE.value: "Do a simple computation.",
    AnswerType.NUMBER.value: "Solve the math world problem, giving your answer as an arabic numeral. Answer with only the final number.",
    AnswerType.TEXT.value: "Solve the problem below.",
}

human_prompts = {
    AnswerType.BOOL.value: 'Answer as "yes" or "no":\n',
    AnswerType.MULTIPLE_CHOICE.value: "MATH WORLD PROBLEM CHOICE (A) (B) (C) (D) or (E):\n",
    AnswerType.NUMBER.value: "Single numeric answer:\n",
    AnswerType.TEXT.value: "Answer as just the letters:\n",
}
existing_file_paths_results = set()
with open(file_path_info_results, 'r', encoding='utf-8') as resultsfile:
    reader = csv.DictReader(resultsfile)
    for row in reader:
        existing_file_paths_results.add(row['result_file_name'])

custom_name_str = f'_{CUSTOM_NAME}' if CUSTOM_NAME else ''
for folder in all_dataset_folders:
    # question_file = f'../datasets/{folder}/data/data_normalized.csv'
    # if not os.path.exists(question_file):
    path_to_question_file = f'../datasets/{folder}/data'
    files_in_folder = FileUtils.get_all_files_and_folders(path_to_question_file, only_folders=False)
    if len(files_in_folder) > 0:
        question_files = FileUtils.select_folders_datasets(files_in_folder)
        question_files_it = [os.path.join(path_to_question_file, q_file) for q_file in question_files]
    else:
        continue
    for question_file in question_files_it:
        print(question_file)
        question_filename = question_file.split('/')[-1].split('\\')[-1].split('.')[0]
        result_file = f'../datasets/{folder}/results/{METHOD_NAME_FILE}_{current_date}_{question_filename}{custom_name_str}.csv'
        # Open the results file in append mode and read existing ids
        existing_ids = set()

        try:
            with open(result_file, 'r', encoding='utf-8') as resultsfile:
                reader = csv.DictReader(resultsfile)
                for row in reader:
                    existing_ids.add(int(row['id']))
        except FileNotFoundError:
            with open(result_file, 'w', newline='', encoding='utf-8') as resultsfile:
                writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                writer.writeheader()


        with open(question_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            with open(result_file, 'a', newline='', encoding='utf-8') as resultsfile:
                writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                for idx, row in tqdm(enumerate(reader), total=TOTAL_COUNT):
                    if idx >= TOTAL_COUNT:
                        break
                    answer_type = row['answer_type']
                    system_prompt = system_prompts[answer_type]
                    choices_str_info = "Choices:\n```\n{row['choices']}\n```\n" if row['choices'] else ''
                    facts_str_info = "Facts:\n```\n{row['facts']}\n```\n" if row['facts'] else ''
                    start_human_prompt_info = 'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else 'Problem:\n```\n{question}\n```\n'
                    end_human_prompt = human_prompts[answer_type]
                    human_prompt_info = f"{start_human_prompt_info}{choices_str_info}{facts_str_info}{end_human_prompt}"
                    if int(row['id']) not in existing_ids:
                        question = row['question']
                        qid = row['quid']
                        answer = row['answer']
                        answer_word = row['answer_word'] if row['answer_word'] != '' else None
                        choices_str = f"Choices:\n```\n{row['choices']}\n```\n" if row['choices'] else ''
                        facts_str = f"Facts:\n```\n{row['facts']}\n```\n" if row['facts'] else ''
                        start_human_prompt = f'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else f'Problem:\n```\n{question}\n```\n'
                        human_prompt = f"{start_human_prompt}{choices_str}{facts_str}{end_human_prompt}"
                        answer_llm_unedited = controller_ai.get_llm_api_response_with_backup_special(
                            system_prompt, human_prompt, response_count=ANSWER_COUNT, temperature=TEMPERATURE,
                            get_multiple_answers=True, model_name=MODEL_NAME
                        )
                        answers_before_processing = '\n------\n'.join(answer_llm_unedited)
                        for answer_idx, answer_generated in enumerate(answer_llm_unedited):
                            answer_llm_unedited[answer_idx] = ResultUtils.preprocess_answer(answer_generated, answer_type)
                        answer_llm = RewardMethods.majority_element(answer_llm_unedited)
                        if answer_llm is not None:
                            correct = ResultUtils.check_corrct_answer(
                                llm_answer=answer_llm, true_answer=answer, other_true_answer=answer_word,
                                answer_type=answer_type
                            )
                        else:
                            correct = False
                        data_results = DataResults(
                            id=int(row['id']),
                            quid=qid,
                            question=question,
                            true_answer=answer,
                            llm_answer=answers_before_processing,
                            correct=correct,
                            llm_answer_chosen=answer_llm,
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
            system_prompt=system_prompt,
            human_prompt=human_prompt_info,
            temperature=TEMPERATURE,
            response_count=ANSWER_COUNT,
            reward_method=REWARD_METHOD,
            llm_model=controller_ai.model.name if MODEL_NAME is None else MODEL_NAME
        )
        info_result.result_file_name = resultsfile.name
        info_result.count = TOTAL_COUNT
        info_result.accuracy = ResultUtils.count_correct_values(resultsfile.name)

        # Convert the dataclass instance to a dictionary
        data_to_append = asdict(info_result)

        # Append the data to the CSV file
        if info_result.result_file_name not in existing_file_paths_results:
            with open(file_path_info_results, 'a', newline='', encoding='utf-8') as csvfile:
                writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
                writer_info.writerow(data_to_append)
        else:  # TODO edit the existing row
            previous_highest_id -= 1
