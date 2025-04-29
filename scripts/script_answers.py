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
from models.Enums.OutputFormat import OutputFormat
from models.constants import human_prompts, system_prompts, mutated_task_prompts_AQuA_RAT, system_prompts_output, \
    system_prompts_static, system_prompts_task, created_my_prompts_MC, created_my_prompts_NUM
from models.DataClass.AnswerResults import AnswerResults
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
import random


CUSTOM_NAME = None#'structured extra'#'always_start_with_same' #None #'answer_update2'#None  # 'all_dataset'#None  # 'NO_REASONING'
TOTAL_COUNT = 1300
TEMPERATURE = 0.0
ANSWER_COUNT = 1
N_SAMPLES_MUT = 5
METHOD: Method = Method.MUT
METHOD_NAME_FILE = str(METHOD)
REWARD_METHOD: RewardMethod = RewardMethod.RERANK #None  # RewardMethod.MAJOR.value
MODEL_NAME = 'gpt-4o'  # 'o3-mini', 'gpt-4o-mini', 'gpt-4o'
PREDEFINED_DATASETS: List[str] | None = [str(Datasets.AQUA.value)]
PREDEFINED_FILES: List[str] | None = None #['data_normalized_STEM_dev.csv', 'data_normalized_STEM_val.csv']
USE_SYSTEM_PROMPT_STRUCTURE = False
MUTATION_UNTIL_SATISFIED = False
OUTPUT_FORMAT = OutputFormat.STRUCTURED_COT
REWARD_NAME = None

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
        """
        Retrieve a set of existing file paths from a CSV file.

        Args:
            file_path_info_results: The path to the CSV file containing the results.

        Returns:
            A set of file paths extracted from the 'result_file_name' column in the CSV file.
        """
        existing_file_paths_results = set()
        with open(file_path_info_results, 'r', encoding='utf-8') as resultsfile:
            reader = csv.DictReader(resultsfile)
            for row in reader:
                existing_file_paths_results.add(row['result_file_name'])
        return existing_file_paths_results

    def get_file_answer_types(self, question_file_path: str) -> Set[str]:
        """
        Extract unique answer types from a CSV file.

        Args:
            question_file_path: The path to the CSV file containing questions and their answer types.

        Returns:
            A set of unique answer types found in the 'answer_type' column of the CSV file.
        """
        # Get unique answer types
        unique_answer_types = set()
        with open(question_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                unique_answer_types.add(row['answer_type'])
        return unique_answer_types

    def iterate_through_folders(self, system_prompt_task: str | None = None):
        """
        Iterate through dataset folders and process question files.

        This method retrieves all dataset folders, filters them based on predefined datasets,
        and processes the question files within each folder. It uses the provided system prompt
        for task initialization.

        Args:
            system_prompt_task: The initial system prompt for the task. If None, a default prompt will be used
        """
        all_dataset_folders = FileUtils.get_all_files_and_folders('../datasets', only_folders=True)
        all_dataset_folders = FileUtils.select_folders_datasets(all_dataset_folders,
                                                                predifined_names=PREDEFINED_DATASETS)
        for folder in all_dataset_folders:
            path_to_question_file = f'../datasets/{folder}/data'
            files_in_folder = FileUtils.get_all_files_and_folders(path_to_question_file, only_folders=False)
            if len(files_in_folder) > 0:
                if len(files_in_folder) > 1:
                    question_files = FileUtils.select_folders_datasets(files_in_folder, predifined_names=PREDEFINED_FILES)
                else:
                    question_files = files_in_folder
                question_files_it = [os.path.join(path_to_question_file, q_file) for q_file in question_files]
            else:
                logger.error(f'No files in {path_to_question_file} folder')
                continue

            self.iterate_through_question_files(question_files_it, folder, system_prompt_task_initial=system_prompt_task)

    def iterate_through_prompts(self, prompts_for_iteration: List[str] | None = None):
        """
        Iterate through a list of prompts and process dataset folders.

        This method processes each prompt in the provided list (or a default list if none is provided),
        logs the prompt index and content, updates the custom name string for each prompt, and iterates
        through dataset folders using the current prompt.

        Args:
            prompts_for_iteration: A list of prompts to iterate through. If None, a default list of mutated task
            prompts is used.
        """
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
            answer_type: AnswerType,
            answer: str,
            answer_word: str,
            result_file: str
    ) -> AnswerResults:
        """
        Generate results for a given question using the specified method and reward strategy.

        This method processes a question and generates results based on the selected method and reward strategy.
        It supports various methods such as zero-shot, structured output, mutation-based approaches, and more.

        Args:
            human_prompt: The input prompt/question for the LLM.
            system_prompt: The system-level instructions for the LLM.
            answer_type: The type of the answer (e.g., MULTIPLE_CHOICE, NUMBER).
            answer: The correct answer for validation.
            answer_word: An optional additional correct answer for validation.
            result_file: The path to the file where results will be saved.

        Returns:
            AnswerResults: An object containing the LLM's output, correctness, and other metadata.

        Raises:
            Exception: If the specified method or reward strategy is not implemented.
        """
        answer_results = AnswerResults()
        try:
            if METHOD == Method.A_2:
                if REWARD_METHOD == RewardMethod.MAJOR or REWARD_METHOD == RewardMethod.RERANK:
                    answer_results = self.controller_answers.get_n_sampling_llm_answer_majority(
                        system_prompt=system_prompt, human_prompt=human_prompt, response_count=ANSWER_COUNT,
                        temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word, reward_method=REWARD_METHOD,
                        output_format=OUTPUT_FORMAT
                    )
                else:
                    raise Exception(f"Reward method not implemented. Method: {METHOD.value}; Reward method: {REWARD_METHOD.value}")
            elif METHOD == Method.A_1 and REWARD_METHOD is None:
                if OUTPUT_FORMAT == OutputFormat.NO_FORMAT:
                    answer_results = self.controller_answers.get_zero_shot_answer(
                        system_prompt=system_prompt, human_prompt=human_prompt,
                        temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word
                    )
                else:
                    answer_results = self.controller_answers.get_structured_output(
                        human_prompt=human_prompt, system_prompt=system_prompt, model_name=MODEL_NAME,
                        response_count=ANSWER_COUNT, temperature=TEMPERATURE, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word, output_format=OUTPUT_FORMAT
                    )
            elif METHOD == Method.MUT:
                if MUTATION_UNTIL_SATISFIED:
                    answer_results = self.controller_answers.get_answer_with_adaptive_mutation(
                        system_prompt=system_prompt, human_prompt=human_prompt, model_name=MODEL_NAME,
                        answer_type=answer_type, ground_truth_answer=answer, result_file=result_file,
                        ground_truth_answer_word=answer_word, temperature=TEMPERATURE, get_structured_output=True
                    )
                else:
                    answer_results = self.controller_answers.get_answer_with_mutation(
                        system_prompt=system_prompt, human_prompt=human_prompt, n_samples=N_SAMPLES_MUT,
                        temperature=TEMPERATURE, model_name=MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word, reward_method=REWARD_METHOD,
                        output_format=OUTPUT_FORMAT
                    )
                if answer_results.task_prompts_chosen is not None:
                    answer_results.task_system_prompts = answer_results.task_prompts_chosen
                # if answer_results.task_prompts_majority is not None and answer_results.task_prompts_majority != "" \
                #         and (METHOD == Method.MUT_M.value or METHOD == Method.STRUCT_MUT_M.value):
                #     answer_results.task_system_prompts = answer_results.task_prompts_majority.split('\n------\n')
                # elif answer_results.task_prompts_correct is not None and answer_results.task_prompts_correct != "" \
                #         and (METHOD == Method.MUT_C.value or METHOD == Method.STRUCT_MUT_C.value):
                #     answer_results.task_system_prompts = answer_results.task_prompts_correct.split('\n------\n')
            elif METHOD == Method.PS or METHOD == Method.PS_PLUS or METHOD == Method.ZS_COT \
                    or METHOD == Method.TWO_PROMPTS:
                if METHOD == Method.PS or METHOD == Method.PS_PLUS or METHOD == Method.ZS_COT:
                    system_prompt = None
                answer_results = self.controller_answers.get_two_prompts_output(
                    question_text=human_prompt, model_name=MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word, temperature=TEMPERATURE,
                    method=METHOD, system_prompt=system_prompt
                )

            else:
                raise Exception(f"Method not implemented. Method: {METHOD.value}; Reward method: {REWARD_METHOD.value}")

        except Exception as e:
            logger.error(e)
        return answer_results

    def get_dictreader_size(self, file_path: str) -> int:
        """
        Calculate the number of rows in a CSV file.

        Args:
            file_path: The path to the CSV file.

        Returns:
            int: The total number of rows in the CSV file.
        """
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
        """
        Process a list of question files within a specified folder.

        This method iterates through a list of question files, extracts answer types, constructs prompts,
        and generates results for each question. It also handles existing results and updates the results file.

        Args:
            question_files_it: A list of file paths to the question files to process.
            folder: The name of the folder containing the question files.
            system_prompt_task_initial: The initial system prompt for the task. If None, a default prompt will be
                                        used based on the answer type.
        """
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
            system_prompt_info_start = system_prompt

            print(question_file)
            question_filename = question_file.split('/')[-1].split('\\')[-1].split('.')[0]
            reward_str = f"_{REWARD_METHOD.value}" if REWARD_METHOD else ''
            # TODO other stuff as well
            result_file = f'../datasets/{folder}/results/{METHOD_NAME_FILE}{reward_str}_' \
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
                        answer_type_str = row['answer_type']
                        answer_type = AnswerType[answer_type_str]

                        if 'MUTATION' in METHOD_NAME_FILE:
                            system_prompt_task = random.choice(task_system_prompts) if len(task_system_prompts) > 0 else system_prompt_task_initial
                            system_prompt = f"{system_prompt_task}\n\n{system_prompts_output[AnswerType.MULTIPLE_CHOICE.value]}\n\n{system_prompts_static[AnswerType.MULTIPLE_CHOICE.value]}"
                            if answer_type != AnswerType.MULTIPLE_CHOICE:
                                raise Exception(f"Answer type {answer_type.value} is not currently supported for mutation "
                                                f"method. Only MULTIPLE_CHOICE is supported.")
                            task_system_prompts = []

                        choices_str_info = "Choices:\n```\n{choices}\n```\n" if row['choices'] else ''
                        facts_str_info = "Facts:\n```\n{facts}\n```\n" if row['facts'] else ''
                        start_human_prompt_info = 'Question:\n```\n{question}\n```\n' if 'question' in system_prompt.lower() else 'Problem:\n```\n{question}\n```\n'
                        end_human_prompt = human_prompts[answer_type.value]
                        human_prompt_info = f"{start_human_prompt_info}{choices_str_info}{facts_str_info}{end_human_prompt}"
                        system_prompt_info = system_prompt

                        if int(row['id']) not in existing_ids:
                            question = row['question']
                            qid = row['quid']
                            answer = row['answer']
                            answer_word = row['answer_word'] if row['answer_word'] != '' else None
                            choices_str = f"Choices:\n```\n{row['choices']}\n```\n" if row['choices'] else ''
                            if answer_word is None and answer_type == AnswerType.MULTIPLE_CHOICE:
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
                            if METHOD == Method.PS or METHOD == Method.PS_PLUS or \
                                    METHOD == Method.ZS_COT:# or METHOD == Method.TWO_PROMPTS:
                                if answer_type == AnswerType.MULTIPLE_CHOICE:
                                    choices_list = row['choices'].split('\n')
                                    human_prompt = f"Q: {question} Answer Choices: {'('+' ('.join(choices_list)}"
                                    human_prompt_info = "Q: {question} Answer Choices: {choices}"
                                else:
                                    human_prompt = f"Q: {question}"
                                    human_prompt_info = "Q: {question}"
                                system_prompt_info = ""
                            answer_results = self.get_results_for_question(
                                human_prompt=human_prompt, system_prompt=system_prompt, answer_type=answer_type,
                                answer=answer, answer_word=answer_word, result_file=result_file
                            )
                            if METHOD == Method.MUT  and len(answer_results.task_system_prompts) > 0:
                                task_system_prompts = answer_results.task_system_prompts
                            # task_prompts_majority_str = answer_results.task_prompts_majority if \
                            #     (METHOD == Method.MUT_Me or METHOD == Method.STRUCT_MUT_M) else \
                            #     (answer_results.task_prompts_correct if (METHOD == Method.MUT_C or
                            #                                              METHOD == Method.STRUCT_MUT_C.) else None)
                            data_results = DataResults(
                                id=int(row['id']),
                                quid=qid,
                                question=question_and_choices,
                                true_answer=answer,
                                llm_answer=answer_results.llm_answer_unedited,
                                correct=answer_results.correct,
                                llm_answer_chosen=answer_results.chosen_answer,
                                reward_method=REWARD_METHOD.value,
                                reward_score=answer_results.score_chosen,
                                task_prompt_all=answer_results.task_prompts_all,
                                task_prompts_majority=str(task_system_prompts),
                            )
                            writer.writerow(asdict(data_results))
            current_result_id = FileUtils.get_highest_id_from_csv(self.file_path_info_all_results) + 1
            if 'MUTATION' in METHOD_NAME_FILE:
                system_prompt_info = system_prompt_info_start
            info_result = InfoResults(
                id=current_result_id,
                date=self.current_date,
                dataset_name=folder,
                method=METHOD.value,
                finished=True,
                system_prompt=system_prompt_info,
                human_prompt=human_prompt_info,
                temperature=TEMPERATURE,
                response_count=ANSWER_COUNT,
                reward_method=REWARD_METHOD.value,
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

        This method logs details about the current execution, including the method, model name, reward method,
        temperature, dataset folder, question file, initial prompt, and result file. It also logs warnings
        and critical messages based on the configuration of the run.

        Args:
            folder: The name of the dataset folder being processed. Defaults to None.
            q_file: The name of the question file being processed. Defaults to None.
            initial_prompt: The initial system prompt used for the task. Defaults to None.
            result_file: The path to the result file where outputs are saved. Defaults to None.

        """
        output_str = ""
        output_str += f"\nMethod: {METHOD.value}"
        output_str += "\t with additional instructions for task prompt" if USE_SYSTEM_PROMPT_STRUCTURE \
            else "\t only task prompt"
        output_str += f"\t\tModel name: {MODEL_NAME if MODEL_NAME else self.controller_answers.controller_ai.model.name}"
        output_str += f"\t\tReward method: {REWARD_METHOD.value}\n" if REWARD_METHOD else "\t\tNo Reward method\n"
        output_str += f"Temperature: {TEMPERATURE}\t\tAnswer count: {ANSWER_COUNT}\t\tSamples for mutation: " \
                      f"{N_SAMPLES_MUT}\t\tMAX dataset size: {TOTAL_COUNT}\n"
        output_str += f"Dataset folder: {folder}\t\tQuestion file: {q_file}\t\tInitial prompt: {initial_prompt}\n"
        output_str += f"Result file: {result_file}\n"
        logger.info(output_str)
        # Some warnings, errors
        if USE_SYSTEM_PROMPT_STRUCTURE and OUTPUT_FORMAT != OutputFormat.NO_FORMAT:
            logger.critical(f"using input prompt structure with structured output format [{USE_SYSTEM_PROMPT_STRUCTURE=}]")
        if METHOD == Method.A_2 and ANSWER_COUNT == 1:
            logger.critical(f"using N_SAMPLES with answer count = 1")
        if METHOD == Method.A_2 and REWARD_METHOD is None:
            logger.critical(f"using N_SAMPLES without a reward method")
        if METHOD == Method.A_2 and TEMPERATURE <= 0.3:
            logger.critical(f"using N_SAMPLES with low temperature [{TEMPERATURE=}]")
        if METHOD in [Method.A_2, Method.MUT] and REWARD_METHOD is None:
            logger.critical(f"using {METHOD} with no reward method")
        if (METHOD in [Method.A_1, Method.PS, Method.PS_PLUS, Method.ZS_COT, Method.TWO_PROMPTS] and
                REWARD_METHOD is not None):
            logger.critical(f"using {METHOD.value} with reward method {REWARD_METHOD.value}")
        if (METHOD.value in [Method.MUT.value, Method.TWO_PROMPTS.value] and USE_SYSTEM_PROMPT_STRUCTURE and
                OUTPUT_FORMAT != OutputFormat.NO_FORMAT):
            logger.critical(f"using {METHOD.value} with system prompt structure")
        if METHOD.value in [Method.PS.value, Method.PS_PLUS.value, Method.ZS_COT.value] and OUTPUT_FORMAT != OutputFormat.NO_FORMAT:
            logger.critical(f"using {METHOD.value} with no output format")


if __name__ == "__main__":

    llm_runner = LLMRunner()
    # llm_runner.iterate_through_prompts()
    llm_runner.iterate_through_folders(system_prompt_task='Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).')



    # TODO ########################################################################################################

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
    # USE_SYSTEM_PROMPT_STRUCTURE = False
    # TEMPERATURE = 0.0
    # ANSWER_COUNT = 1
    # REWARD_METHOD = None
    # prompts_for_iteration = mutated_task_prompts_AQuA_RAT + created_my_prompts_MC
    # llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)
    #
    # # # TODO (not comment) STRUCT_EXTRA = 'STRUCTURED_EXTRA'  # Structured output with explanation and extra
    # METHOD = Method.STRUCT_EXTRA.value
    # METHOD_NAME_FILE = METHOD
    # logger.info(f"Method: {METHOD}")
    # USE_SYSTEM_PROMPT_STRUCTURE = False
    # TEMPERATURE = 0.0
    # ANSWER_COUNT = 1
    # REWARD_METHOD = None
    # OUTPUT_FORMAT = OutputFormat.STRUCTURED_EXTRA
    # CUSTOM_NAME = 'RETRY_F'
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_MC)
    #
    # # # TODO (not comment) STRUCT_ANS = 'STRUCTURED_ONLY_ANSWER'
    # METHOD = Method.STRUCT_ANS.value
    # CUSTOM_NAME = None
    # METHOD_NAME_FILE = METHOD
    # logger.info(f"Method: {METHOD}")
    # USE_SYSTEM_PROMPT_STRUCTURE = False
    # TEMPERATURE = 0.0
    # ANSWER_COUNT = 1
    # REWARD_METHOD = None
    # OUTPUT_FORMAT = OutputFormat.STRUCTURED_ANSWER
    # prompts_for_iteration = mutated_task_prompts_AQuA_RAT + created_my_prompts_MC
    # llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

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