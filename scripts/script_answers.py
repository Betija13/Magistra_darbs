import csv
import os
from loguru import logger
from typing import List, Set
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
from models.constants import human_prompts, system_prompts, mutated_task_prompts_AQuA_RAT, system_prompts_output, \
    system_prompts_static, system_prompts_task, created_my_prompts_MC, created_my_prompts_NUM
from models.DataClass.AnswerResults import AnswerResults
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
import random


CUSTOM_NAME = None  # 'all_dataset'#None  # 'NO_REASONING'
TOTAL_COUNT = 1000
TEMPERATURE = 0.0
ANSWER_COUNT = 1
N_SAMPLES_MUT = 5
METHOD: str = str(Method.STRUCT_MUT_M.value)
METHOD_NAME_FILE = METHOD
REWARD_METHOD = RewardMethod.MAJOR.value #None  # RewardMethod.MAJOR.value
MODEL_NAME = 'gpt-4o'  # 'o3-mini', 'gpt-4o-mini', 'gpt-4o'
PREDIFINED_DATASETS: List[str] | None = [str(Datasets.THEOREMAQA.value)]
PREDIFINED_FILES: List[str] | None = None #['data_normalized_test.csv']
USE_SYSTEM_PROMPT_STRUCTURE = False


class LLMRunner:
    def __init__(self, controller_answers: AnswerMethods | None = None):
        if controller_answers:
            self.controller_answers = controller_answers
        else:
            self.controller_answers = AnswerMethods()
        self.folder = '../datasets'
        self.filename_all_results = 'info_results.csv'
        self.file_path_info_all_results = os.path.join(self.folder, self.filename_all_results)
        self.fieldnames = [field.name for field in DataResults.__dataclass_fields__.values()]
        self.current_date = datetime.now().strftime('%d-%m-%Y')
        self.custom_name_str = f'_{CUSTOM_NAME}' if CUSTOM_NAME else ''

    def get_existing_file_paths_results(self, file_path_info_results: str) -> Set[str]:
        existing_file_paths_results = set()
        with open(file_path_info_results, 'r', encoding='utf-8') as resultsfile:
            reader = csv.DictReader(resultsfile)
            for row in reader:
                existing_file_paths_results.add(row['result_file_name'])
        return existing_file_paths_results

    def get_file_answer_types(self, question_file_path: str) -> Set[str]:
        # Get unique answer types
        unique_answer_types = set()
        with open(question_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                unique_answer_types.add(row['answer_type'])
        return unique_answer_types

    def iterate_through_folders(self, system_prompt_task: str | None = None):
        all_dataset_folders = FileUtils.get_all_files_and_folders('../datasets', only_folders=True)
        all_dataset_folders = FileUtils.select_folders_datasets(all_dataset_folders,
                                                                predifined_names=PREDIFINED_DATASETS)
        for folder in all_dataset_folders:
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

            self.iterate_through_question_files(question_files_it, folder, system_prompt_task_initial=system_prompt_task)

    def iterate_through_prompts(self, prompts_for_iteration: List[str]|None = None):
        if prompts_for_iteration is None:
            prompts_for_iteration = mutated_task_prompts_AQuA_RAT
        for prompt_idx, system_prompt_task in enumerate(prompts_for_iteration):
            logger.info(f'Prompt index: {prompt_idx}; Prompt: {system_prompt_task}')
            self.custom_name_str = f'_{CUSTOM_NAME}_{prompt_idx}' if CUSTOM_NAME else f'_{prompt_idx}'
            self.iterate_through_folders(system_prompt_task=system_prompt_task)

    def get_results_for_question(
            self,
            human_prompt: str,
            system_prompt: str,
            answer_type: str,
            answer: str,
            answer_word: str
    ) -> AnswerResults:
        answer_results = AnswerResults()
        try:
            if METHOD == Method.A_2.value and REWARD_METHOD == RewardMethod.MAJOR.value:
                answer_results = self.controller_answers.get_n_sampling_llm_answer_majority(
                    system_prompt=system_prompt, human_prompt=human_prompt, response_count=ANSWER_COUNT,
                    temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word
                )
            elif METHOD == Method.A_1.value and REWARD_METHOD is None:
                answer_results = self.controller_answers.get_zero_shot_answer(
                    system_prompt=system_prompt, human_prompt=human_prompt,
                    temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word
                )
            elif (METHOD == Method.MUT_M.value or METHOD == Method.MUT_C.value or
                  METHOD == Method.STRUCT_MUT_M.value or METHOD == Method.STRUCT_MUT_C.value) and \
                    REWARD_METHOD == RewardMethod.MAJOR.value:
                answer_results = self.controller_answers.get_answer_with_mutation(
                    system_prompt=system_prompt, human_prompt=human_prompt, n_samples=N_SAMPLES_MUT,
                    temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word, get_structured_output=METHOD.startswith('STRUCTURED')
                )
                if answer_results.task_prompts_majority is not None and answer_results.task_prompts_majority != "" \
                        and (METHOD == Method.MUT_M.value or METHOD == Method.STRUCT_MUT_M.value):
                    answer_results.task_system_prompts = answer_results.task_prompts_majority.split('\n------\n')
                elif answer_results.task_prompts_correct is not None and answer_results.task_prompts_correct != "" \
                        and (METHOD == Method.MUT_C.value or METHOD == Method.STRUCT_MUT_C.value):
                    answer_results.task_system_prompts = answer_results.task_prompts_correct.split('\n------\n')
            elif METHOD == Method.STRUCT.value or METHOD == Method.STRUCT_ANS.value:
                only_answer = METHOD == Method.STRUCT_ANS.value
                answer_results = self.controller_answers.get_structured_output(
                    human_prompt=human_prompt, system_prompt=system_prompt, model_name=MODEL_NAME,
                    response_count=ANSWER_COUNT, temperature=TEMPERATURE, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word, only_answer=only_answer
                )
            elif METHOD == Method.PS.value or METHOD == Method.PS_PLUS.value or METHOD == Method.ZS_COT.value \
                    or METHOD == Method.TWO_PROMPTS.value:
                answer_results = self.controller_answers.get_two_prompts_output(
                    question_text=human_prompt, model_name=MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word, temperature=TEMPERATURE,
                    method=METHOD, system_prompt=system_prompt
                )
            elif METHOD == Method.MUT_E.value:
                raise Exception("Mutation Edit method not implemented")
                # TODO add Mutation Edit
            else:
                raise Exception(f"Method not implemented. Method: {METHOD}; Reward method: {REWARD_METHOD}")

        except Exception as e:
            logger.error(e)
        return answer_results

    def get_dictreader_size(self, file_path: str) -> int:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            return len(rows)

    def iterate_through_question_files(
            self,
            question_files_it: List[str],
            folder: str,
            system_prompt_task_initial: str | None = None
    ):
        for question_file in question_files_it:

            answer_types_q_file = self.get_file_answer_types(question_file)
            if len(answer_types_q_file) > 1:
                logger.warning(f'More than one answer type in {question_file}')
                continue
            else:
                question_file_answer_type = list(answer_types_q_file)[0]
            if system_prompt_task_initial is None:
                system_prompt_task_initial = system_prompts_task[question_file_answer_type]
            if USE_SYSTEM_PROMPT_STRUCTURE:
                system_prompt = f"{system_prompt_task_initial}\n\n" \
                                    f"{system_prompts_output[question_file_answer_type]}\n\n" \
                                    f"{system_prompts_static[question_file_answer_type]}"
            else:
                system_prompt = system_prompt_task_initial

            print(question_file)
            question_filename = question_file.split('/')[-1].split('\\')[-1].split('.')[0]
            result_file = f'../datasets/{folder}/results/{METHOD_NAME_FILE}_' \
                          f'{self.current_date}_{question_filename}{self.custom_name_str}.csv'
            self.output_info(
                q_file=question_file, folder=folder, initial_prompt=system_prompt_task_initial, result_file=result_file
            )
            # Open the results file in append mode and read existing ids
            existing_ids = set()
            system_prompt_info = None
            human_prompt_info = None

            try:
                with open(result_file, 'r', encoding='utf-8') as resultsfile:
                    logger.warning(f"File {result_file} already exists")
                    reader = csv.DictReader(resultsfile)
                    for row in reader:
                        existing_ids.add(int(row['id']))
            except FileNotFoundError:
                with open(result_file, 'w', newline='', encoding='utf-8') as resultsfile:
                    writer = csv.DictWriter(resultsfile, fieldnames=self.fieldnames)
                    writer.writeheader()

            with open(question_file, 'r', encoding='utf-8') as csvfile:
                task_system_prompts = []
                reader = csv.DictReader(csvfile)
                size_reader = self.get_dictreader_size(question_file)
                size_iteration_objects = size_reader if size_reader < TOTAL_COUNT else TOTAL_COUNT
                with open(result_file, 'a', newline='', encoding='utf-8') as resultsfile:
                    writer = csv.DictWriter(resultsfile, fieldnames=self.fieldnames)
                    for idx, row in tqdm(enumerate(reader), total=size_iteration_objects):
                        if idx >= TOTAL_COUNT:
                            break
                        answer_type = row['answer_type']

                        if METHOD == Method.MUT_M.value or METHOD == Method.MUT_C.value or \
                                METHOD == Method.STRUCT_MUT_M.value or METHOD == Method.STRUCT_MUT_C.value:
                            system_prompt_task = random.choice(task_system_prompts) if len(task_system_prompts) > 0 else system_prompt_task_initial
                            system_prompt = f"{system_prompt_task}\n\n{system_prompts_output[AnswerType.MULTIPLE_CHOICE.value]}\n\n{system_prompts_static[AnswerType.MULTIPLE_CHOICE.value]}"
                            task_system_prompts = []

                        choices_str_info = "Choices:\n```\n{choices}\n```\n" if row['choices'] else ''
                        facts_str_info = "Facts:\n```\n{facts}\n```\n" if row['facts'] else ''
                        start_human_prompt_info = 'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else 'Problem:\n```\n{question}\n```\n'
                        end_human_prompt = human_prompts[answer_type]
                        human_prompt_info = f"{start_human_prompt_info}{choices_str_info}{facts_str_info}{end_human_prompt}"
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
                            start_human_prompt = f'Problem:\n```\n{question}\n```\n' if 'problem' in system_prompt.lower() else f'Question:\n```\n{question}\n```\n'
                            human_prompt = f"{start_human_prompt}{choices_str}{facts_str}{end_human_prompt}"
                            if METHOD == Method.PS.value or METHOD == Method.PS_PLUS.value or \
                                    METHOD == Method.ZS_COT.value:# or METHOD == Method.TWO_PROMPTS.value:
                                if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                                    choices_list = row['choices'].split('\n')
                                    human_prompt = f"Q: {question} Answer Choices: {'('+' ('.join(choices_list)}"
                                    human_prompt_info = "Q: {question} Answer Choices: {choices}"
                                else:
                                    human_prompt = f"Q: {question}"
                                    human_prompt_info = "Q: {question}"
                                system_prompt_info = ""
                            answer_results = self.get_results_for_question(
                                human_prompt=human_prompt, system_prompt=system_prompt, answer_type=answer_type,
                                answer=answer, answer_word=answer_word
                            )
                            if (METHOD == Method.MUT_M.value or METHOD == Method.MUT_C.value or
                                METHOD == Method.STRUCT_MUT_M.value or METHOD == Method.STRUCT_MUT_C.value) and \
                                    len(answer_results.task_system_prompts) > 0:
                                task_system_prompts = answer_results.task_system_prompts
                            task_prompts_majority_str = answer_results.task_prompts_majority if \
                                (METHOD == Method.MUT_M.value or METHOD == Method.STRUCT_MUT_M.value) else \
                                (answer_results.task_prompts_correct if (METHOD == Method.MUT_C.value or
                                                                         METHOD == Method.STRUCT_MUT_C.value) else None)
                            data_results = DataResults(
                                id=int(row['id']),
                                quid=qid,
                                question=question_and_choices,
                                true_answer=answer,
                                llm_answer=answer_results.llm_answer_unedited,
                                correct=answer_results.correct,
                                llm_answer_chosen=answer_results.chosen_answer,
                                reward_method=REWARD_METHOD,
                                task_prompt_all=answer_results.task_prompts_all,
                                task_prompts_majority=task_prompts_majority_str,
                            )
                            writer.writerow(asdict(data_results))
            current_result_id = FileUtils.get_highest_id_from_csv(self.file_path_info_all_results) + 1
            info_result = InfoResults(
                id=current_result_id,
                date=self.current_date,
                dataset_name=folder,
                method=METHOD,
                finished=True,
                system_prompt=system_prompt_info,
                human_prompt=human_prompt_info,
                temperature=TEMPERATURE,
                response_count=ANSWER_COUNT,
                reward_method=REWARD_METHOD,
                llm_model=self.controller_answers.controller_ai.model.name if MODEL_NAME is None else MODEL_NAME,
            )
            info_result.result_file_name = resultsfile.name
            info_result.count = size_iteration_objects
            numeric_results = ResultUtils.count_correct_values(resultsfile.name)
            info_result.accuracy = numeric_results.accuracy_score
            info_result.percentage_of_short_answers = numeric_results.percentage_of_short_answers

            # Convert the dataclass instance to a dictionary
            data_to_append = asdict(info_result)

            existing_file_paths_results = self.get_existing_file_paths_results(self.file_path_info_all_results)
            # Append the data to the CSV file
            if info_result.result_file_name not in existing_file_paths_results:
                with open(self.file_path_info_all_results, 'a', newline='', encoding='utf-8') as csvfile:
                    writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
                    writer_info.writerow(data_to_append)
            else:
                pass
                # TODO edit the existing row

    def output_info(
            self,
            folder: str | None = None,
            q_file: str | None = None,
            initial_prompt: str | None = None,
            result_file: str | None = None
    ) -> None:
        """
        Output information about the current run.
        """
        output_str = ""
        output_str += f"\nMethod: {METHOD}"
        output_str += "\t with additional instructions for task prompt" if USE_SYSTEM_PROMPT_STRUCTURE \
            else "\t only task prompt"
        output_str += f"\t\tModel name: {MODEL_NAME if MODEL_NAME else self.controller_answers.controller_ai.model.name}"
        output_str += f"\t\tReward method: {REWARD_METHOD}\n" if REWARD_METHOD else "\t\tNo Reward method\n"
        output_str += f"Temperature: {TEMPERATURE}\t\tAnswer count: {ANSWER_COUNT}\t\tSamples for mutation: " \
                      f"{N_SAMPLES_MUT}\t\tMAX dataset size: {TOTAL_COUNT}\n"
        output_str += f"Dataset folder: {folder}\t\tQuestion file: {q_file}\t\tInitial prompt: {initial_prompt}\n"
        output_str += f"Result file: {result_file}\n"
        logger.info(output_str)


if __name__ == "__main__":

    llm_runner = LLMRunner()
    llm_runner.iterate_through_prompts()

    # TODO (not comment) going through all dataset
    # # TODO (not comment) Task prompt only
    # USE_SYSTEM_PROMPT_STRUCTURE = False
    # TEMPERATURE = 0.0
    # ANSWER_COUNT = 1
    # REWARD_METHOD = None
    # METHOD = Method.A_1.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)

    # TODO (not comment) Task prompt +
    # USE_SYSTEM_PROMPT_STRUCTURE = True
    # TEMPERATURE = 0.0
    # ANSWER_COUNT = 1
    # REWARD_METHOD = None
    # CUSTOM_NAME = 'TASK_PROMPT_PLUS'
    # METHOD = Method.A_1.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)

    # # TODO (not comment) STRUCT = 'STRUCTURED_OUTPUT'  # Structured output with explanation
    # METHOD = Method.STRUCT.value
    # METHOD_NAME_FILE = METHOD
    # logger.info(f"Method: {METHOD}")
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)
    #
    # # TODO (not comment) STRUCT_ANS = 'STRUCTURED_ONLY_ANSWER'
    # METHOD = Method.STRUCT_ANS.value
    # METHOD_NAME_FILE = METHOD
    # logger.info(f"Method: {METHOD}")
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)

    # # TODO (not comment) Two prompts
    # METHOD = Method.TWO_PROMPTS.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Plan and solve
    # METHOD = Method.PS.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Plan and solve plus
    # METHOD = Method.PS_PLUS.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Zero shot chain of thought
    # METHOD = Method.ZS_COT.value
    # METHOD_NAME_FILE = METHOD
    # llm_runner.iterate_through_folders()

    # TODO (not comment) MUTATION majority
    # # best task prompt
    # start_prompt = 'Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best task + prompt
    # start_prompt = 'Solve the multiple choice math word problem. Clearly explain each step of your solution process before choosing (A), (B), (C), (D), or (E) as the final answer.'
    # llm_runner.custom_name_str = 'BEST_TASK_PLUS'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best structured cot prompt (skipping - same as best task only prompt)
    #
    # # best structured just answer prompt
    # llm_runner.custom_name_str = 'BEST_STR_ANSW'
    # start_prompt = "To dissect the mystery and make it as obvious as a neon sign in the dark, pretend you're explaining the issue to a bewildered squirrel from another dimension. This interdimensional viewpoint can shed light on the obscure details or universal energies involved. Now, let's solve the multiple-choice math puzzle by selecting one of the intergalactic runes: (A), (B), (C), (D), or (E)."
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best two prompts prompt
    # llm_runner.custom_name_str = 'BEST_TWO_PROMPTS'
    # start_prompt = 'Pick a letter and pray that math agrees with you.'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)

    # TODO (not comment) Mutation correct
    # METHOD = str(Method.STRUCT_MUT_C.value)
    # METHOD_NAME_FILE = METHOD
    # # best task prompt
    # llm_runner.custom_name_str = 'BEST_JUST_TASK'
    # start_prompt = 'Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best task + prompt
    # start_prompt = 'Solve the multiple choice math word problem. Clearly explain each step of your solution process before choosing (A), (B), (C), (D), or (E) as the final answer.'
    # llm_runner.custom_name_str = 'BEST_TASK_PLUS'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best structured cot prompt (skipping - same as best task only prompt)
    #
    # # best structured just answer prompt
    # llm_runner.custom_name_str = 'BEST_STR_ANSW'
    # start_prompt = "To dissect the mystery and make it as obvious as a neon sign in the dark, pretend you're explaining the issue to a bewildered squirrel from another dimension. This interdimensional viewpoint can shed light on the obscure details or universal energies involved. Now, let's solve the multiple-choice math puzzle by selecting one of the intergalactic runes: (A), (B), (C), (D), or (E)."
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)
    #
    # # best two prompts prompt
    # start_prompt = 'Pick a letter and pray that math agrees with you.'
    # llm_runner.custom_name_str = 'BEST_TWO_PROMPTS'
    # llm_runner.iterate_through_folders(system_prompt_task=start_prompt)