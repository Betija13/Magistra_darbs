import csv
import os
from loguru import logger
from typing import List, Set
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
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
from models.Enums.RewardModelNames import RewardModelNames
from models.Enums.Tasks import Tasks
from models.constants import human_prompts, system_prompts, mutated_task_prompts_AQuA_RAT, system_prompts_output, \
    system_prompts_static, system_prompts_task, created_my_prompts_MC, created_my_prompts_NUM, \
    mutated_task_prompts_MMLU, best_task_prompts_MMLU, reward_names_shorten, prompt_names_list
from models.DataClass.AnswerResults import AnswerResults
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict, dataclass, field, replace, fields
import random
import argparse

@dataclass
class Args:
    CUSTOM_NAME: str | None = None
    TOTAL_COUNT: int = 1300
    TEMPERATURE: float = 0.0
    ANSWER_COUNT: int = 1
    N_SAMPLES_MUT: int = 5
    METHOD: Method = Method.MUT
    METHOD_NAME_FILE: str = str(METHOD.value)
    REWARD_METHOD: RewardMethod | None = None# RewardMethod.LLM_O_R #None  # RewardMethod.MAJOR.value
    REWARD_NAME: RewardModelNames | None = None #RewardModelNames.LLM_GEMINI
    MODEL_NAME: str = 'gpt-4o'  # 'o3-mini', 'gpt-4o-mini', 'gpt-4o'
    PREDEFINED_DATASETS: List[str] | None = field(default_factory=lambda: [str(Datasets.MMLU.value)])
    PREDEFINED_FILES: List[str] | None = field(default_factory=lambda:['data_normalized_STEM_dev.csv']) #['data_normalized_STEM_dev.csv', 'data_normalized_STEM_val.csv']
    USE_SYSTEM_PROMPT_STRUCTURE: bool = False
    MUTATION_UNTIL_SATISFIED: bool = False
    OUTPUT_FORMAT: OutputFormat = OutputFormat.STRUCTURED_COT
    TASK: Tasks = Tasks.NOT_CHOSEN
    ORIGINAL_FILE: str | None = None
    SAME_START: bool = False # Mutation always start from the same point
    # SAME_MID_START: bool = False # Mutation in the middle always starts from same
    USE_EXAMPLE_MUT: bool = False
    PROMPTS_ITERATION: List[str] | None = None
    GET_DYNAMIC_SCORES_Q_T: bool = False # When doing mutation, get scores for each answer and mutated task prompt # TODO add to errors
    MUTATE_MUT: bool = False



args = Args()

# TODO finish
# TODO add help everywhere
parser = argparse.ArgumentParser()
parser.add_argument(
    '--REWARD_NAME_VALUE',
    choices=[e.value for e in RewardModelNames]
)
parser.add_argument(
    '--TASK_VALUE', type=str, required=True,
    choices=[e.value for e in Tasks]
)
parser.add_argument(
    '--REWARD_METHOD_VALUE', type=str,
    choices=[e.value for e in RewardMethod]
)
parser.add_argument('--ORIGINAL_FILE', type=str)
parser.add_argument('--PROMPTS_ITERATION', nargs='*')
parser.add_argument(
    '--PROMPT_LIST_EXISTING',
    choices=prompt_names_list.keys()
)
# parser.add_argument('--CUSTOM_NAME', type=str)
# parser.add_argument('--TOTAL_COUNT', type=int)
# parser.add_argument('--TEMPERATURE', type=float)
# parser.add_argument('--ANSWER_COUNT', type=int)

cli_args = parser.parse_args()
if cli_args.REWARD_NAME_VALUE is not None:
    cli_args.REWARD_NAME = RewardModelNames(cli_args.REWARD_NAME_VALUE)
if cli_args.TASK_VALUE is not None:
    cli_args.TASK = Tasks(cli_args.TASK_VALUE)
if cli_args.REWARD_METHOD_VALUE is not None:
    cli_args.REWARD_METHOD = RewardMethod(cli_args.REWARD_METHOD_VALUE)
if cli_args.PROMPT_LIST_EXISTING is not None:
    prompt_iteration_choice = prompt_names_list[cli_args.PROMPT_LIST_EXISTING]
    if cli_args.PROMPTS_ITERATION is None:
        cli_args.PROMPTS_ITERATION = prompt_iteration_choice
    else:
        cli_args.PROMPTS_ITERATION.extend(prompt_iteration_choice)
#
valid_fields = {f.name for f in fields(Args)}
filtered_args = {k: v for k, v in vars(cli_args).items() if v is not None and k in valid_fields}

args = replace(args, **filtered_args)


class LLMRunner:
    def __init__(self, controller_answers: AnswerMethods | None = None):
        if controller_answers:
            self.controller_answers = controller_answers
        else:
            self.controller_answers = AnswerMethods(reward_name=args.REWARD_NAME)
        self.folder = '../datasets'
        self.filename_all_results = 'info_results.csv'
        self.file_path_info_all_results = os.path.join(self.folder, self.filename_all_results)
        self.fieldnames = [field_n.name for field_n in DataResults.__dataclass_fields__.values()]
        self.current_date = datetime.now().strftime('%d-%m-%Y')
        self.custom_name_str = f'_{args.CUSTOM_NAME}' if args.CUSTOM_NAME else ''

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
                                                                predifined_names=args.PREDEFINED_DATASETS)
        for folder in all_dataset_folders:
            path_to_question_file = f'../datasets/{folder}/data'
            files_in_folder = FileUtils.get_all_files_and_folders(path_to_question_file, only_folders=False)
            if len(files_in_folder) > 0:
                if len(files_in_folder) > 1:
                    question_files = FileUtils.select_folders_datasets(files_in_folder, predifined_names=args.PREDEFINED_FILES)
                else:
                    question_files = files_in_folder
                question_files_it = [os.path.join(path_to_question_file, q_file) for q_file in question_files]
            else:
                logger.error(f'No files in {path_to_question_file} folder')
                continue

            self.iterate_through_question_files(question_files_it, folder, system_prompt_task_initial=system_prompt_task)

    def iterate_through_prompts(self, prompts_for_iteration: List[str]):
        """
        Iterate through a list of prompts and process dataset folders.

        This method processes each prompt in the provided list (or a default list if none is provided),
        logs the prompt index and content, updates the custom name string for each prompt, and iterates
        through dataset folders using the current prompt.

        Args:
            prompts_for_iteration: A list of prompts to iterate through. If None, a default list of mutated task
            prompts is used.
        """
        for prompt_idx, system_prompt_task in enumerate(prompts_for_iteration):
            logger.info(f'Prompt index: {prompt_idx}; Prompt: {system_prompt_task}')
            self.custom_name_str = f'_{args.CUSTOM_NAME}_{prompt_idx}' if args.CUSTOM_NAME else f'_{prompt_idx}'
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
            if args.METHOD == Method.A_2:
                if args.REWARD_METHOD == RewardMethod.MAJOR or args.REWARD_METHOD == RewardMethod.RERANK:
                    answer_results = self.controller_answers.get_n_sampling_llm_answer_majority(
                        system_prompt=system_prompt, human_prompt=human_prompt, response_count=args.ANSWER_COUNT,
                        temperature=args.TEMPERATURE, model_name=args.MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word, reward_method=args.REWARD_METHOD,
                        output_format=args.OUTPUT_FORMAT
                    )
                else:
                    raise Exception(f"Reward method not implemented. Method: {args.METHOD.value}; Reward method: {args.REWARD_METHOD.value}")
            elif args.METHOD == Method.A_1 and args.REWARD_METHOD is None:
                if args.OUTPUT_FORMAT == OutputFormat.NO_FORMAT:
                    answer_results = self.controller_answers.get_zero_shot_answer(
                        system_prompt=system_prompt, human_prompt=human_prompt,
                        temperature=args.TEMPERATURE, model_name=args.MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word
                    )
                else:
                    answer_results = self.controller_answers.get_structured_output(
                        human_prompt=human_prompt, system_prompt=system_prompt, model_name=args.MODEL_NAME,
                        response_count=args.ANSWER_COUNT, temperature=args.TEMPERATURE, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word,
                        output_format=args.OUTPUT_FORMAT
                    )
            elif args.METHOD == Method.MUT:
                if args.MUTATION_UNTIL_SATISFIED:
                    if args.REWARD_METHOD == RewardMethod.CORRECT_A:
                        answer_results = self.controller_answers.get_answer_with_adaptive_mutation(
                            system_prompt=system_prompt, human_prompt=human_prompt, model_name=args.MODEL_NAME,
                            answer_type=answer_type, ground_truth_answer=answer, result_file=result_file,
                            ground_truth_answer_word=answer_word, temperature=args.TEMPERATURE,
                            output_format=args.OUTPUT_FORMAT, use_example_mut=args.USE_EXAMPLE_MUT,
                            get_mut_scores=args.GET_DYNAMIC_SCORES_Q_T
                        )
                    else:
                        raise Exception(f"Reward method {args.REWARD_METHOD.value if args.REWARD_METHOD else None} not implementedl")
                else:
                    answer_results = self.controller_answers.get_answer_with_mutation(
                        system_prompt=system_prompt, human_prompt=human_prompt, n_samples=args.N_SAMPLES_MUT,
                        temperature=args.TEMPERATURE, model_name=args.MODEL_NAME, answer_type=answer_type,
                        ground_truth_answer=answer, ground_truth_answer_word=answer_word,
                        reward_method=args.REWARD_METHOD, output_format=args.OUTPUT_FORMAT,
                        use_example_mut=args.USE_EXAMPLE_MUT, mutate_mutation=args.MUTATE_MUT
                    )
                if answer_results.task_prompts_chosen is not None and len(answer_results.task_prompts_chosen) > 0:
                    answer_results.task_system_prompts = answer_results.task_prompts_chosen
            elif args.METHOD == Method.PS or args.METHOD == Method.PS_PLUS or args.METHOD == Method.ZS_COT \
                    or args.METHOD == Method.TWO_PROMPTS:
                if args.METHOD == Method.PS or args.METHOD == Method.PS_PLUS or args.METHOD == Method.ZS_COT:
                    system_prompt = None
                answer_results = self.controller_answers.get_two_prompts_output(
                    question_text=human_prompt, model_name=args.MODEL_NAME, answer_type=answer_type,
                    ground_truth_answer=answer, ground_truth_answer_word=answer_word, temperature=args.TEMPERATURE,
                    method=args.METHOD, system_prompt=system_prompt
                )

            else:
                raise Exception(f"Method not implemented. Method: {args.METHOD.value}; Reward method: {args.REWARD_METHOD.value}")

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
            if args.USE_SYSTEM_PROMPT_STRUCTURE:
                system_prompt = f"{system_prompt_task_initial}\n\n" \
                                    f"{system_prompts_output[question_file_answer_type]}\n\n" \
                                    f"{system_prompts_static[question_file_answer_type]}"
            else:
                system_prompt = system_prompt_task_initial
            system_prompt_info_start = system_prompt

            question_filename = question_file.split('/')[-1].split('\\')[-1].split('.')[0]
            reward_str = f"__{args.REWARD_METHOD.value}" if args.REWARD_METHOD else ''
            until_satisfied_mut_str = f"__CONT-US" if args.MUTATION_UNTIL_SATISFIED else ''
            same_start_str = f"__SAME-S" if args.SAME_START else \
                ("__DIF-S" if args.MUTATION_UNTIL_SATISFIED else '')
            use_example_mut = f"__USE-EXM" if args.USE_EXAMPLE_MUT else \
                ("__NO-EXM" if args.MUTATION_UNTIL_SATISFIED else '')
            input_format = '_I-S_' if args.USE_SYSTEM_PROMPT_STRUCTURE else '_I-N_'
            mut_mut_str = ""
            if args.METHOD == Method.MUT and not args.MUTATION_UNTIL_SATISFIED:
                mut_mut_str = "__MM" if args.MUTATE_MUT else "__MT"

            result_file = (f'../datasets/{folder}/results/{args.METHOD_NAME_FILE}{until_satisfied_mut_str}{same_start_str}{use_example_mut}{mut_mut_str}{reward_str}__'
                           f'O-{args.OUTPUT_FORMAT.value}__{input_format}{self.current_date}__F-{question_filename}'
                           f'{self.custom_name_str}.csv')
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
                size_iteration_objects = size_reader if size_reader < args.TOTAL_COUNT else args.TOTAL_COUNT
                with open(result_file, 'a', newline='', encoding='utf-8') as resultsfile:
                    writer = csv.DictWriter(resultsfile, fieldnames=self.fieldnames)
                    for idx, row in tqdm(enumerate(reader), total=size_iteration_objects):
                        if idx >= args.TOTAL_COUNT:
                            break
                        answer_type_str = row['answer_type']
                        answer_type = AnswerType[answer_type_str]

                        if args.METHOD == Method.MUT:
                            system_prompt_task = random.choice(task_system_prompts) if len(task_system_prompts) > 0 else system_prompt_task_initial
                            if args.USE_SYSTEM_PROMPT_STRUCTURE:
                                system_prompt = f"{system_prompt_task}\n\n{system_prompts_output[AnswerType.MULTIPLE_CHOICE.value]}\n\n{system_prompts_static[AnswerType.MULTIPLE_CHOICE.value]}"
                            else:
                                system_prompt = system_prompt_task
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
                            if args.METHOD == Method.PS or args.METHOD == Method.PS_PLUS or args.METHOD == Method.ZS_COT:# or args.METHOD == Method.TWO_PROMPTS:
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
                            if args.METHOD == Method.MUT and len(answer_results.task_system_prompts) > 0:
                                task_system_prompts = answer_results.task_system_prompts
                                if args.SAME_START:
                                    task_system_prompts = []
                            data_results = DataResults(
                                id=int(row['id']),
                                quid=qid,
                                question=question_and_choices,
                                true_answer=answer,
                                llm_answer=answer_results.llm_answer_unedited,
                                correct=answer_results.correct,
                                llm_answer_chosen=answer_results.chosen_answer,
                                reward_method=args.REWARD_METHOD.value if args.REWARD_METHOD else None,
                                reward_score=answer_results.score_chosen,
                                task_prompt_all=answer_results.task_prompts_all,
                                task_prompts_majority=str(task_system_prompts),
                            )
                            writer.writerow(asdict(data_results))
            current_result_id = FileUtils.get_highest_id_from_csv(self.file_path_info_all_results) + 1
            if args.METHOD == Method.MUT:
                system_prompt_info = system_prompt_info_start
            info_result = InfoResults(
                id=current_result_id,
                date=self.current_date,
                dataset_name=folder,
                method=args.METHOD.value,
                finished=True,
                system_prompt=system_prompt_info,
                human_prompt=human_prompt_info,
                temperature=args.TEMPERATURE,
                response_count=args.ANSWER_COUNT,
                reward_method=args.REWARD_METHOD.value if args.REWARD_METHOD else None,
                llm_model=self.controller_answers.controller_ai.model.name if args.MODEL_NAME is None else args.MODEL_NAME,
                reward_name=args.REWARD_NAME.value if args.REWARD_NAME else None
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
        and critical messages based on the configuration of the run. If there are any errors, the code will be exited.

        Args:
            folder: The name of the dataset folder being processed. Defaults to None.
            q_file: The name of the question file being processed. Defaults to None.
            initial_prompt: The initial system prompt used for the task. Defaults to None.
            result_file: The path to the result file where outputs are saved. Defaults to None.

        """
        has_error = False
        output_str = ""
        output_str += f"\nMethod: {args.METHOD.value}"
        output_str += "\t with additional instructions for task prompt" if args.USE_SYSTEM_PROMPT_STRUCTURE \
            else "\t only task prompt"
        output_str += f"\t\tModel name: {args.MODEL_NAME if args.MODEL_NAME else self.controller_answers.controller_ai.model.name}"
        output_str += f"\t\tReward method: {args.REWARD_METHOD.value}\n" if args.REWARD_METHOD else "\t\tNo Reward method\n"
        output_str += f"Temperature: {args.TEMPERATURE}\t\tAnswer count: {args.ANSWER_COUNT}\t\tSamples for mutation: " \
                      f"{args.N_SAMPLES_MUT}\t\tMAX dataset size: {args.TOTAL_COUNT}\n"
        output_str += f"Dataset folder: {folder}\t\tQuestion file: {q_file}\t\tInitial prompt: {initial_prompt}\n"
        output_str += f"Result file: {result_file}\n"
        logger.info(output_str)
        # Some warnings, errors
        if args.USE_SYSTEM_PROMPT_STRUCTURE and args.OUTPUT_FORMAT != OutputFormat.NO_FORMAT:
            logger.critical(f"using input prompt structure with structured output format [{args.USE_SYSTEM_PROMPT_STRUCTURE=}]")
            has_error = True
        if args.METHOD == Method.A_2 and args.ANSWER_COUNT == 1:
            logger.critical(f"using N_SAMPLES with answer count = 1")
            has_error = True
        if args.METHOD == Method.A_2 and args.TEMPERATURE <= 0.3:
            logger.critical(f"using N_SAMPLES with low temperature [{args.TEMPERATURE=}]")
            has_error = True
        if args.METHOD in [Method.A_2, Method.MUT] and args.REWARD_METHOD is None:
            logger.critical(f"using {args.METHOD.value} with no reward method")
            has_error = True
        if (args.METHOD in [Method.A_1, Method.PS, Method.PS_PLUS, Method.ZS_COT, Method.TWO_PROMPTS] and
                args.REWARD_METHOD is not None):
            logger.critical(f"using {args.METHOD.value} with reward method {args.REWARD_METHOD.value}")
            has_error = True
        if (args.METHOD in [Method.MUT, Method.TWO_PROMPTS] and args.USE_SYSTEM_PROMPT_STRUCTURE and
                args.OUTPUT_FORMAT != OutputFormat.NO_FORMAT):
            logger.critical(f"using {args.METHOD.value} with system prompt structure")
            has_error = True
        if args.METHOD in [Method.PS, Method.PS_PLUS, Method.ZS_COT] and args.OUTPUT_FORMAT != OutputFormat.NO_FORMAT:
            logger.critical(f"using {args.METHOD.value} with no output format")
            has_error = True
        if ((args.REWARD_METHOD == RewardMethod.REWARD_M and
            args.REWARD_NAME not in [RewardModelNames.INTERNLM_1_8_B, RewardModelNames.INTERNLM_7_B,
                                     RewardModelNames.DEBERTA_V3_2, RewardModelNames.BLENDER_PRM,
                                     RewardModelNames.SAFAIRXC, RewardModelNames.QRM_8B_V2, RewardModelNames.QRM_8B,
                                     RewardModelNames.QRM_8B_P, RewardModelNames.GRM, RewardModelNames.SKYWORK,
                                     RewardModelNames.ARMORM, RewardModelNames.URM, RewardModelNames.PAIRRM_B,
                                     RewardModelNames.EURUS, RewardModelNames.GRM_3B, RewardModelNames.GRM_2B])
            or ((args.REWARD_METHOD == RewardMethod.LLM_O_R or args.REWARD_METHOD == RewardMethod.LLM_O_B_I) and
                args.REWARD_NAME != RewardModelNames.LLM_GEMINI)
            or (args.REWARD_METHOD == RewardMethod.RERANK and args.REWARD_NAME != RewardModelNames.RERANK_MODEL)
            or (args.REWARD_METHOD == RewardMethod.MAJOR and args.REWARD_NAME is not None)
            or (args.REWARD_METHOD == RewardMethod.CORRECT_A and args.REWARD_NAME is not None)
        ):
            logger.critical(f"Reward name not part of reward method. Reward method: {args.REWARD_METHOD}\t"
                            f"Reward name: {args.REWARD_NAME}")
            has_error = True
        if args.MUTATION_UNTIL_SATISFIED and args.METHOD != Method.MUT:
            logger.critical(f"MUTATION_UNTIL_SATISFIED chosen with non mutation method {args.METHOD.value}")
            has_error = True
        if args.METHOD == Method.MUT and args.N_SAMPLES_MUT < 2:
            logger.critical(f"N_SAMPLES_MUT chosen {args.N_SAMPLES_MUT} for mutation method (should be at least 2, preferably 5).")
            has_error = True
        if has_error:
            logger.critical(f"Something wrong with arguments. ^ Please fix that.")
            exit()

    def go_through_all_existing(self) -> None:
        """
        Goes through all existing methods (Plan and solve, plan and solve plus, zero shot chain of thought).
        """
        args.TEMPERATURE = 0.0
        args.ANSWER_COUNT = 1
        args.REWARD_METHOD = None
        args.REWARD_NAME = None
        args.USE_SYSTEM_PROMPT_STRUCTURE = False
        args.OUTPUT_FORMAT = OutputFormat.NO_FORMAT

        args.METHOD = Method.PS
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        llm_runner.iterate_through_folders()

        args.METHOD = Method.PS_PLUS
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        llm_runner.iterate_through_folders()

        args.METHOD = Method.ZS_COT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        llm_runner.iterate_through_folders()

    def go_through_all_static_zero_shot(self, prompts_for_iteration: List[str]) -> None:
        """
        Goes through all static zero-shot methods.
        Args:
            prompts_for_iteration: Prompts tht will be tested.

        """
        args.TEMPERATURE = 0.0
        args.ANSWER_COUNT = 1
        args.REWARD_METHOD = None
        args.REWARD_NAME = None

        # only task prompt
        args.METHOD = Method.A_1
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.USE_SYSTEM_PROMPT_STRUCTURE = False
        args.OUTPUT_FORMAT = OutputFormat.NO_FORMAT
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        # task prompt plus
        args.USE_SYSTEM_PROMPT_STRUCTURE = True
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        # structured cot
        args.OUTPUT_FORMAT = OutputFormat.STRUCTURED_COT
        args.USE_SYSTEM_PROMPT_STRUCTURE = False
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        # structured only answer
        args.OUTPUT_FORMAT = OutputFormat.STRUCTURED_ANSWER
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        # two prompts
        args.METHOD = Method.TWO_PROMPTS
        args.OUTPUT_FORMAT = OutputFormat.NO_FORMAT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def go_through_static_n_shot(
            self,
            prompts_for_iteration: List[str],
            temperature: float | None = None,
            answer_count: int | None = None,
            reward_method: RewardMethod | None = None,
            reward_name: RewardModelNames | None = None
    ) -> None:
        """
        Goes through static n-shot method.
        Args:
            prompts_for_iteration: Prompts tht will be tested.
            temperature: Temperature for the model (0-2). For this method, recommended to be at least 0.3.
            answer_count: Number of answers to generate (n-shot count).
            reward_method: Reward method to use.
            reward_name: Name of the reward model if reward method is a model.

        """

        args.TEMPERATURE = 1.0 if temperature is None else temperature
        args.ANSWER_COUNT = 5 if answer_count is None else answer_count
        args.METHOD = Method.A_2
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.REWARD_METHOD = RewardMethod.MAJOR if reward_method is None else reward_method
        args.REWARD_NAME = None if reward_name is None else reward_name
        args.USE_SYSTEM_PROMPT_STRUCTURE = False
        args.OUTPUT_FORMAT = OutputFormat.STRUCTURED_COT

        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def go_through_dynamic_until_correct_all(self, prompts_for_iteration: List[str]) -> None:
        """
        Goes through all dynamic mutation method variants with chosen prompts.
        Args:
            prompts_for_iteration: Prompts tht will be tested.
        """
        args.METHOD= Method.MUT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.REWARD_METHOD = RewardMethod.CORRECT_A
        args.REWARD_NAME = None
        args.MUTATION_UNTIL_SATISFIED = True
        args.TEMPERATURE = 0.0
        args.ANSWER_COUNT = 1
        args.N_SAMPLES_MUT = 0

        args.SAME_START = False
        args.USE_EXAMPLE_MUT = False
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        args.SAME_START = False
        args.USE_EXAMPLE_MUT = True
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        args.SAME_START = True
        args.USE_EXAMPLE_MUT = False
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        args.SAME_START = True
        args.USE_EXAMPLE_MUT = True
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def go_through_dynamic_until_correct_no_example(self, prompts_for_iteration: List[str]) -> None:
        """
        Goes through dynamic mutation method variants that do not use example for mutation with chosen prompts.
        Args:
            prompts_for_iteration: Prompts tht will be tested.
        """
        args.METHOD= Method.MUT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.REWARD_METHOD = RewardMethod.CORRECT_A
        args.REWARD_NAME = None
        args.MUTATION_UNTIL_SATISFIED = True
        args.TEMPERATURE = 0.0
        args.ANSWER_COUNT = 1
        args.N_SAMPLES_MUT = 0
        args.USE_EXAMPLE_MUT = False

        args.SAME_START = False
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

        args.SAME_START = True
        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def go_through_dynamic_n_shot_mut_task(
            self,
            prompts_for_iteration: List[str],
            temperature: float | None = None,
            n_samples_mut: int | None = None
    ) -> None:
        """
        Goes through dynamic mutation with selected reward method. Mutates only task prompt.
        Args:
            prompts_for_iteration: Prompts tht will be tested.
            temperature: Temperature (0-2)
            n_samples_mut: Mutation examples for each iteration.
        Returns:

        """
        args.METHOD = Method.MUT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.MUTATION_UNTIL_SATISFIED = False
        args.SAME_START = False
        args.USE_EXAMPLE_MUT = False
        args.TEMPERATURE = 1.0 if temperature is None else temperature
        args.ANSWER_COUNT = 1
        args.N_SAMPLES_MUT = 5 if n_samples_mut is None else n_samples_mut
        args.MUTATE_MUT = False
        args.USE_EXAMPLE_MUT = False

        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def go_through_dynamic_n_shot_mut_mut(
            self,
            prompts_for_iteration: List[str],
            temperature: float | None = None,
            n_samples_mut: int | None = None
    ):
        """
        Goes through dynamic mutation with selected reward method. Mutates mutated task prompt.
        Args:
            prompts_for_iteration: Prompts tht will be tested.
            temperature: Temperature (0-2)
            n_samples_mut: Mutation examples for each iteration.
        Returns:

        """
        args.METHOD = Method.MUT
        args.METHOD_NAME_FILE = str(args.METHOD.value)
        args.MUTATION_UNTIL_SATISFIED = False
        args.SAME_START = False
        args.USE_EXAMPLE_MUT = False
        args.TEMPERATURE = 1.0 if temperature is None else temperature
        args.ANSWER_COUNT = 1
        args.N_SAMPLES_MUT = 5 if n_samples_mut is None else n_samples_mut
        args.MUTATE_MUT = True
        args.USE_EXAMPLE_MUT = False

        llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    def get_n_samples_different_scores(
            self,
            original_file_path: str,
            reward_methods: List[RewardMethod],
            reward_name: RewardModelNames | None = None
    ):
        """
        Goes through existing N-sample file and rates answers by different method.
        Args:
            original_file_path: Original n-sample file path.
            reward_methods: list of used reward methods.
            reward_name: Reward model name.
        """
        # checks if original file exists, otherwise error
        if not os.path.exists(original_file_path):
            logger.error(f"Path {original_file_path} does not exist")
        else:
            all_reward_methods = [m.value for m in RewardMethod]
            reward_name_str = None
            for reward_method in reward_methods:
                try:
                    reward_method_str = reward_method.value
                    if reward_method == RewardMethod.REWARD_M and reward_name is not None:
                        self.controller_answers.initialize_reward_model(reward_name)
                    if reward_method == RewardMethod.REWARD_M and reward_name is None:
                        if args.REWARD_NAME is None:
                            logger.error("Chosen reward method without a reward name")
                        else:
                            reward_name = args.REWARD_NAME
                    if reward_name is not None:
                        reward_name_str = reward_name.value
                    if reward_method == RewardMethod.RERANK:
                        reward_name = RewardModelNames.RERANK_MODEL
                    if reward_method == RewardMethod.LLM_O_R or reward_method == RewardMethod.LLM_O_B_I:
                        reward_name = RewardModelNames.LLM_GEMINI
                    if reward_method == RewardMethod.MAJOR:
                        reward_method = None
                    original_reward_method = [existing_meth for existing_meth in all_reward_methods if existing_meth in  original_file_path]
                    if len(original_reward_method) == 1:
                        original_reward_method = original_reward_method[0]
                    else:
                        raise Exception(f"Weird original reward method. {original_reward_method}")
                    output_file = original_file_path
                    output_file = output_file.replace(original_reward_method, reward_method_str)
                    if reward_name_str is not None and reward_method == RewardMethod.REWARD_M:
                        if output_file.endswith('.csv'):
                            custom_rn = [cname for r, cname in reward_names_shorten if r == reward_name][0]
                            output_file = f"{output_file[:-4]}__RM-{custom_rn}{output_file[-4:]}"
                        else:
                            raise Exception(f"Weird output file: {output_file}")

                    info_results_object = None
                    with open(self.file_path_info_all_results, 'r', encoding='utf-8') as csv_file:
                        reader = csv.DictReader(csv_file)
                        for row in reader:
                            if row['result_file_name'] == original_file_path:
                                if info_results_object is not None:
                                    logger.error("Info row is already something")
                                info_results_object = InfoResults(**row)
                            if row['result_file_name'] == output_file:
                                raise Exception(f"File {output_file} already in info file")
                    info_results_object.date = self.current_date
                    info_results_object.reward_method = str(reward_method.value)
                    info_results_object.reward_name = reward_name.value if reward_name is not None else None
                    info_results_object.result_file_name = output_file
                    existing_ids = set()
                    fieldnames = [field_n.name for field_n in DataResults.__dataclass_fields__.values()]
                    try:
                        with open(output_file, 'r', encoding='utf-8') as resultsfile:
                            logger.warning(f"File {output_file} already exists")
                            reader = csv.DictReader(resultsfile)
                            for row in reader:
                                existing_ids.add(int(row['id']))
                    except FileNotFoundError:
                        with open(output_file, 'w', newline='', encoding='utf-8') as resultsfile:
                            writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                            writer.writeheader()
                    updated_rows = []
                    with open(output_file, 'a', newline='', encoding='utf-8') as resultsfile:
                        writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                        with open(original_file_path, 'r', encoding='utf-8') as csv_file:
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
                                if reward_method == RewardMethod.LLM_O_R:
                                    answer_obj = self.controller_answers.reward_methods.get_llm_best_answer_reranker(
                                        question=question, answer_options=cot_parts
                                    )
                                    if answer_obj is not None:
                                        try:
                                            score = answer_obj.answer_score
                                            chosen_answer = answer_obj.chosen_answer
                                            chosen_idx = cot_parts.index(
                                                chosen_answer) if chosen_answer is not None else None
                                        except Exception as e:
                                            logger.error(e)
                                            logger.exception(e)
                                            score = None
                                            chosen_answer = None
                                            chosen_idx = None
                                elif reward_method == RewardMethod.LLM_O_B_I:
                                    answer_obj = self.controller_answers.reward_methods.get_llm_best_answer_best_idx(
                                        question=question, answer_options=cot_parts
                                    )
                                    if answer_obj is not None:
                                        try:
                                            score = answer_obj.answer_score
                                            chosen_answer = answer_obj.chosen_answer
                                            chosen_idx = cot_parts.index(
                                                chosen_answer) if chosen_answer is not None else None
                                        except Exception as e:
                                            logger.error(e)
                                            logger.exception(e)
                                            score = None
                                            chosen_answer = None
                                            chosen_idx = None
                                elif reward_method == RewardMethod.REWARD_M:
                                    answer_obj = self.controller_answers.reward_methods.get_reward_model_best_answer(
                                        question=question, answer_options=cot_parts
                                    )
                                    if answer_obj is not None:
                                        try:
                                            score = answer_obj.answer_score
                                            chosen_answer = answer_obj.chosen_answer
                                            chosen_idx = cot_parts.index(
                                                chosen_answer) if chosen_answer is not None else None
                                        except Exception as e:
                                            logger.error(e)
                                            logger.exception(e)
                                            score = None
                                            chosen_answer = None
                                            chosen_idx = None
                                elif reward_method == RewardMethod.MAJOR:
                                    score = None
                                    chosen_answer_letter = RewardMethods.majority_element(final_answer_parts)
                                    chosen_idx = final_answer_parts.index(
                                        chosen_answer_letter) if chosen_answer_letter is not None else None
                                elif reward_method == RewardMethod.RERANK:
                                    answer_obj = self.controller_answers.reward_methods.get_reranking_model_best_answer(
                                        question=question, answer_options=cot_parts
                                    )
                                    score = answer_obj.answer_score
                                    chosen_answer = answer_obj.chosen_answer
                                    chosen_idx = cot_parts.index(chosen_answer)
                                else:
                                    raise Exception(f"method {reward_method} not implemented!!!")
                                final_str = answers_list[chosen_idx] if chosen_idx is not None else None
                                final_letter_answer = final_answer_parts[chosen_idx] if chosen_idx is not None else None
                                # chosen_answer = row['llm_answer_chosen']
                                true_answer = row['true_answer']
                                correct_answer = False
                                if final_letter_answer == true_answer:
                                    correct_answer = True
                                row['llm_answer_chosen'] = final_str
                                # row['llm_answer_chosen'] = final_letter_answer
                                row['reward_score'] = score if chosen_idx is not None else None
                                row['reward_method'] = reward_method.value
                                row['correct'] = correct_answer
                                updated_rows.append(row)
                                writer.writerow(row)
                    current_result_id = FileUtils.get_highest_id_from_csv(self.file_path_info_all_results) + 1
                    info_results_object.id = current_result_id
                    numeric_results = ResultUtils.count_correct_values(output_file)
                    info_results_object.accuracy = numeric_results.accuracy_score
                    info_results_object.percentage_of_short_answers = numeric_results.percentage_of_short_answers

                    data_to_append = asdict(info_results_object)

                    with open(self.file_path_info_all_results, 'a', newline='', encoding='utf-8') as csvfile:
                        writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
                        writer_info.writerow(data_to_append)
                    logger.success(f"Successful {reward_method} \t {reward_name}")

                except Exception as e:
                    logger.error(e)


if __name__ == "__main__":

    llm_runner = LLMRunner()
    if args.TASK == Tasks.N_SAMPLE_DIFF_SCORE_RM:
        llm_runner.get_n_samples_different_scores(
            original_file_path=args.ORIGINAL_FILE,
            reward_methods=[RewardMethod.REWARD_M], reward_name=None)
        llm_runner.controller_answers.reward_methods.delete_rm_file()
    elif args.TASK == Tasks.N_SAMPLE_DIFF_SCORE_OTHER:
        llm_runner.get_n_samples_different_scores(
            original_file_path=args.ORIGINAL_FILE,
            reward_methods=[RewardMethod.RERANK, RewardMethod.LLM_O_R, RewardMethod.LLM_O_B_I, RewardMethod.MAJOR],
            reward_name=None)
    elif args.TASK == Tasks.STATIC_METHODS:
         llm_runner.go_through_all_static_zero_shot(prompts_for_iteration=args.PROMPTS_ITERATION)
        # TODO implement by each method
    elif args.TASK == Tasks.EXISTING_M:
        llm_runner.go_through_all_existing()
        # TODO implement by each method
    elif args.TASK == Tasks.N_SAMPLES_FIRST:
         llm_runner.go_through_static_n_shot(prompts_for_iteration=args.PROMPTS_ITERATION)
        # TODO implement by each method
    elif args.TASK == Tasks.DYNAMIC_CORRECT_ALL:
         llm_runner.go_through_dynamic_until_correct_all(prompts_for_iteration=args.PROMPTS_ITERATION)
    elif args.TASK == Tasks.DYNAMIC_CORRECT_NO_EXAMPLE:
         llm_runner.go_through_dynamic_until_correct_no_example(prompts_for_iteration=args.PROMPTS_ITERATION)
        # TODO implement by each method
    elif args.TASK == Tasks.DYNAMIC_REWARD_MT:
        llm_runner.go_through_dynamic_n_shot_mut_task(prompts_for_iteration=args.PROMPTS_ITERATION)
    elif args.TASK == Tasks.DYNAMIC_REWARD_MM:
        llm_runner.go_through_dynamic_n_shot_mut_mut(prompts_for_iteration=args.PROMPTS_ITERATION)
        # TODO implement by each method
    elif args.TASK == Tasks.MANUAL_ITERATE:
        logger.error(f"Method not yet implemented in script")



    # llm_runner.go_through_static_n_shot(prompts_for_iteration=best_task_prompts_MMLU)
    # llm_runner.iterate_through_prompts()
    # llm_runner.iterate_through_folders(system_prompt_task='Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).')



    # TODO ########################################################################################################

    # TODO (not comment) going through all dataset
    # # TODO (not comment) Task prompt only
    # args.USE_SYSTEM_PROMPT_STRUCTURE = False
    # args.TEMPERATURE = 0.0
    # args.ANSWER_COUNT = 1
    # args.REWARD_METHOD = None
    # args.METHOD = Method.A_1.value
    # args.METHOD_NAME_FILE = args.METHOD
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)

    # TODO (not comment) Task prompt +
    # args.USE_SYSTEM_PROMPT_STRUCTURE = True
    # args.TEMPERATURE = 0.0
    # args.ANSWER_COUNT = 1
    # args.REWARD_METHOD = None
    # CUSTOM_NAME = 'TASK_PROMPT_PLUS'
    # args.METHOD = Method.A_1.value
    # args.METHOD_NAME_FILE = args.METHOD
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_NUM)

    # # TODO (not comment) STRUCT = 'STRUCTURED_OUTPUT'  # Structured output with explanation
    # args.METHOD = Method.STRUCT.value
    # args.METHOD_NAME_FILE = args.METHOD
    # logger.info(f"Method: {args.METHOD}")
    # args.USE_SYSTEM_PROMPT_STRUCTURE = False
    # args.TEMPERATURE = 0.0
    # args.ANSWER_COUNT = 1
    # args.REWARD_METHOD = None
    # prompts_for_iteration = mutated_task_prompts_AQuA_RAT + created_my_prompts_MC
    # llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)
    #
    # # # TODO (not comment) STRUCT_EXTRA = 'STRUCTURED_EXTRA'  # Structured output with explanation and extra
    # args.METHOD = Method.STRUCT_EXTRA.value
    # args.METHOD_NAME_FILE = args.METHOD
    # logger.info(f"Method: {args.METHOD}")
    # args.USE_SYSTEM_PROMPT_STRUCTURE = False
    # args.TEMPERATURE = 0.0
    # args.ANSWER_COUNT = 1
    # args.REWARD_METHOD = None
    # args.OUTPUT_FORMAT = OutputFormat.STRUCTURED_EXTRA
    # CUSTOM_NAME = 'RETRY_F'
    # llm_runner.iterate_through_prompts(prompts_for_iteration=created_my_prompts_MC)
    #
    # # # TODO (not comment) STRUCT_ANS = 'STRUCTURED_ONLY_ANSWER'
    # args.METHOD = Method.STRUCT_ANS.value
    # CUSTOM_NAME = None
    # args.METHOD_NAME_FILE = args.METHOD
    # logger.info(f"Method: {args.METHOD}")
    # args.USE_SYSTEM_PROMPT_STRUCTURE = False
    # args.TEMPERATURE = 0.0
    # args.ANSWER_COUNT = 1
    # args.REWARD_METHOD = None
    # args.OUTPUT_FORMAT = OutputFormat.STRUCTURED_ANSWER
    # prompts_for_iteration = mutated_task_prompts_AQuA_RAT + created_my_prompts_MC
    # llm_runner.iterate_through_prompts(prompts_for_iteration=prompts_for_iteration)

    # # TODO (not comment) Two prompts
    # args.METHOD = Method.TWO_PROMPTS.value
    # args.METHOD_NAME_FILE = args.METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Plan and solve
    # args.METHOD = Method.PS.value
    # args.METHOD_NAME_FILE = args.METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Plan and solve plus
    # args.METHOD = Method.PS_PLUS.value
    # args.METHOD_NAME_FILE = args.METHOD
    # llm_runner.iterate_through_folders()

    # # TODO (not comment) Zero shot chain of thought
    # args.METHOD = Method.ZS_COT.value
    # args.METHOD_NAME_FILE = args.METHOD
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
    # args.METHOD = str(Method.STRUCT_MUT_C.value)
    # args.METHOD_NAME_FILE = args.METHOD
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