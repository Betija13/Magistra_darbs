from typing import List
import re
from loguru import logger
from dataclasses import asdict
import csv
import os
from controllers.AiLLM import ControllerAiLLM
from controllers.RewardMethods import RewardMethods
from controllers.Mutation import Mutation, my_mutation_prompts, my_thinking_styles
from utils.result_utils import ResultUtils
from utils.file_utils import FileUtils
from models.constants import system_prompts_output, system_prompts_static
from models.DataClass.AnswerResults import AnswerResults
from models.DataClass.StructuredOutput import StructuredOutput
from models.DataClass.DynamicMutationInfo import DynamicMutationInfo
from models.DataClass.AnswerExtraction import AnswerExtraction
from models.DataClass.StructuredOutput import StructuredOutput
from models.Enums.RewardMethod import RewardMethod
from models.Enums.AnswerType import AnswerType
from models.Enums.Method import Method
from models.Enums.Datasets import Datasets
from models.Enums.OutputFormat import OutputFormat
from models.Enums.RewardModelNames import RewardModelNames


class AnswerMethods:
    def __init__(self, reward_name: RewardModelNames | None = None):
        self.controller_ai = ControllerAiLLM()
        self.controller_mutation = Mutation(controller_ai=self.controller_ai)
        self.reward_methods = RewardMethods(ai_llm=self.controller_ai)
        if reward_name and reward_name not in [RewardModelNames.LLM_GEMINI, RewardModelNames.RERANK_MODEL]:
            self.initialize_reward_model(reward_name=reward_name)
        self.last_mutation_prompt_idx: int = 0
        self.last_thinking_style_idx: int = 0
        self.max_mutation_prompt_idx: int = len(my_mutation_prompts) - 1
        self.max_thinking_style_idx: int = len(my_thinking_styles) - 1

    def initialize_reward_model(self, reward_name: RewardModelNames) -> None:
        """
        Initializes the reward model.
        Args:
            reward_name: Name of reward model.
        """
        self.reward_methods.init_reward_model(reward_model_name=reward_name)

    def get_zero_shot_answer(
            self,
            system_prompt: str,
            human_prompt: str,
            temperature: float,
            model_name: str | None,
            answer_type: AnswerType,
            ground_truth_answer: str,
            ground_truth_answer_word: str | None = None
    ) -> AnswerResults:
        """
        Get the answer using zero-shot method from the API and check if it is correct or not.
        Args:
            system_prompt: System prompt.
            human_prompt: Human prompt.
            temperature: Temperature (0-2).
            model_name: Name of used LLM.
            answer_type: Type of answer. Enum from AnswerType.
            ground_truth_answer: Ground truth answer.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).

        Returns:
            AnswerResults, which contains the LLM output and if the answer was correct or not.

        """
        answer_results = AnswerResults()
        try:
            answer_llm = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=human_prompt, response_count=1, temperature=temperature,
                get_multiple_answers=False, model_name=model_name
            )[0]
            answer_results.llm_answer_unedited = answer_llm
            answer_llm = ResultUtils.preprocess_answer(answer_llm, answer_type)
            correct = ResultUtils.check_corrct_answer(
                llm_answer=answer_llm, true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                answer_type=answer_type
            )
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def get_n_sampling_llm_answer_majority(
            self,
            system_prompt: str,
            human_prompt: str,
            response_count: int,
            temperature: float,
            model_name: str | None,
            answer_type: AnswerType,
            reward_method: RewardMethod,
            output_format: OutputFormat,
            ground_truth_answer: str,
            ground_truth_answer_word: str | None = None
    ) -> AnswerResults:
        """
        Get the answer using N-sampling method from the API and check if it is correct or not. Correct answer is
        chosen by majority, meaning, the answer that appears the most.
        Args:
            system_prompt: System prompt.
            human_prompt: Human prompt.
            response_count: Count of N responses.
            temperature: Temperature (0-2).
            model_name: Name of used LLM.
            answer_type: Type of answer. Enum from AnswerType.
            reward_method: Reward method to use to get llms answer.
            output_format: Output format of llms answer.
            ground_truth_answer: Ground truth answer.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).

        Returns:
            AnswerResults, which contains the LLM output and if the answer was correct or not.
        """
        answer_results = AnswerResults()
        try:
            answer_llm_unedited = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=human_prompt, response_count=response_count,
                temperature=temperature, get_multiple_answers=True, model_name=model_name,
                output_format=output_format, answer_type=answer_type
            )
            answers_for_voting = []
            if output_format == OutputFormat.NO_FORMAT:
                answers_before_processing = '\n------\n'.join(answer_llm_unedited)
                answer_results.llm_answer_unedited = answers_before_processing
                for answer_idx, answer_generated in enumerate(answer_llm_unedited):
                    processed_answer = ResultUtils.preprocess_answer(answer_generated, answer_type)
                    answers_for_voting.append(processed_answer)
            else:
                if output_format == OutputFormat.STRUCTURED_COT:
                    answers_for_voting = []
                    answer_strs = []
                    final_answers = []
                    for answer_n in answer_llm_unedited:
                        if not isinstance(answer_n, StructuredOutput):
                            raise Exception(f"Answer is not structured output. Answer: {answer_n}")
                        cot_part = answer_n.solution_explanation
                        final_answer = self.get_final_answer_from_structure(
                            answer_type=answer_type, answer_llm_structure=answer_n
                        )
                        final_answers.append(final_answer)
                        answer_str = self.get_str_from_structure(
                            answer_type=answer_type, answer_llm_structure=answer_n
                        )
                        answer_strs.append(answer_str)
                        if reward_method == RewardMethod.MAJOR:
                            answers_for_voting.append(final_answer)
                        else:
                            answers_for_voting.append(cot_part)
                    answers_llm_all = '\n------\n'.join(answer_strs)
                    answer_results.llm_answer_unedited = answers_llm_all

            if reward_method == RewardMethod.MAJOR:
                chosen_answer = RewardMethods.majority_element(answers_for_voting)
            elif reward_method == RewardMethod.RERANK:
                chosen_answer_obj = self.reward_methods.get_reranking_model_best_answer(
                    question=human_prompt, answer_options=answers_for_voting
                )
                chosen_answer = chosen_answer_obj.chosen_answer
                score_answer = chosen_answer_obj.answer_score
                answer_results.score_chosen = score_answer
            if output_format != OutputFormat.NO_FORMAT:
                idx_chosen = answers_for_voting.index(chosen_answer)
                str_chosen = answer_strs[idx_chosen]
                final_chosen_answer = final_answers[idx_chosen]
            else:
                str_chosen = chosen_answer
                final_chosen_answer = chosen_answer

            answer_results.chosen_answer = str_chosen
            if final_chosen_answer is not None:
                correct = ResultUtils.check_corrct_answer(
                    llm_answer=str(final_chosen_answer), true_answer=ground_truth_answer,
                    other_true_answer=ground_truth_answer_word, answer_type=answer_type
                )
            else:
                correct = False
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def get_str_from_structure(self, answer_type: AnswerType, answer_llm_structure: StructuredOutput) -> str | None:
        """
        Get the string from the structured output.
        Args:
            answer_type: Answer type from AnswerType Enum.
            answer_llm_structure: Answer as StructuredOutput object.

        Returns:
            string answer from StructuredOutput object.
        """
        str_answer = None
        try:
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                final_answer = answer_llm_structure.answer_as_letter
                final_answer_str = f"ANSWER_AS_LETTER: {final_answer}"
            elif answer_type == AnswerType.NUMBER:
                final_answer = answer_llm_structure.answer_as_number
                final_answer_str = f"ANSWER_AS_NUMBER: {final_answer}"
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
            cot_part = answer_llm_structure.solution_explanation
            str_answer = f"SOLUTION_EXPLANATION: {cot_part}\n{final_answer_str}"
        except Exception as e:
            logger.error(e)
        return str_answer

    def get_final_answer_from_structure(self, answer_type: AnswerType, answer_llm_structure: StructuredOutput) -> str | None:
        """
        Get the final answer (letter, number) from the structured output.
        Args:
            answer_type: Answer type from AnswerType Enum.
            answer_llm_structure: Answer as StructuredOutput object.

        Returns:
            Final answer as string from StructuredOutput object.
        """
        final_answer = None
        try:
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                final_answer = answer_llm_structure.answer_as_letter
            elif answer_type == AnswerType.NUMBER:
                final_answer = answer_llm_structure.answer_as_number
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
        except Exception as e:
            logger.error(e)
        return final_answer

    def get_answer_with_mutation(
            self,
            system_prompt: str,
            human_prompt: str,
            n_samples: int,
            temperature: float,
            model_name: str | None,
            answer_type: AnswerType,
            ground_truth_answer: str,
            ground_truth_answer_word: str | None = None,
            reward_method: RewardMethod = RewardMethod.MAJOR,
            output_format: OutputFormat = OutputFormat.NO_FORMAT,
            use_example_mut: bool = False,
            mutate_mutation: bool = False
    ) -> AnswerResults:
        """
        Gets n_samples of answers but each answer has different system prompt. The system prompt is mutated.
        Args:
            system_prompt: System prompt.
            human_prompt: Human prompt.
            n_samples: Count of iterated mutations.
            temperature: Temperature (0-2).
            model_name: Name of used LLM.
            answer_type: Type of answer. Enum from AnswerType.
            ground_truth_answer: Ground truth answer.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).
            reward_method: Reward method to choose llm answer.
            output_format: Output type from llm.
            use_example_mut: If true, uses question and answer example for task prompt mutation.
            mutate_mutation: If true, mutates mutated task prompt, if False, mutates only task prompt.

        Returns:
            AnswerResults, which contains the LLM output and if the answer was correct or not.
        """
        answer_results = AnswerResults()
        try:
            answers_llm_unedited = []
            answer_for_voting = []
            final_answers = []
            task_prompts = [system_prompt.split('\n\n')[0]]
            if use_example_mut:
                example = f"{human_prompt}\nCorrect answer (desired output): ```{ground_truth_answer}```"
            else:
                example = None
            if mutate_mutation:
                prompt_for_mutation = system_prompt.split('\n\n')[0]
                for i in range(n_samples-1):
                    mutated_task_prompt = self.controller_mutation.mutate_current_prompt(
                        n_mutations=1, prompt_for_mutation=prompt_for_mutation, output_example=example,
                        use_my_mut_think=True
                    )
                    if mutated_task_prompt != "":
                        task_prompts.append(mutated_task_prompt)
                        prompt_for_mutation = mutated_task_prompt
            else:
                mutated_task_prompts = self.controller_mutation.mutate_source_prompt(
                    n_samples=n_samples-1, prompt_for_mutation=system_prompt.split('\n\n')[0], output_example=example,
                    use_my_mut_think=True
                )
                for task_prompt_n in mutated_task_prompts:
                    if task_prompt_n != "":
                        task_prompts.append(task_prompt_n)
            for task_prompt_n in task_prompts:
                system_prompt = f"{task_prompt_n}\n\n{system_prompts_output[answer_type.value]}\n\n{system_prompts_static[answer_type.value]}"
                if output_format != OutputFormat.NO_FORMAT:
                    system_prompt = task_prompt_n
                    answer_llm_structure = self.controller_ai.get_structured_output(
                        system_prompt=system_prompt, human_prompt=human_prompt, response_count=1,
                        temperature=temperature, model_name=model_name, answer_type=answer_type
                    )
                    final_answer = self.get_final_answer_from_structure(
                        answer_type=answer_type, answer_llm_structure=answer_llm_structure[0]
                    )
                    answer_llm_unedited = self.get_str_from_structure(
                        answer_type=answer_type, answer_llm_structure=answer_llm_structure[0]
                    )
                    cot_part = answer_llm_structure[0].solution_explanation
                else:
                    answer_llm_unedited = self.controller_ai.get_llm_api_response_with_backup_special(
                        system_prompt=system_prompt, prompt=human_prompt, response_count=1, temperature=temperature,
                        get_multiple_answers=False, model_name=model_name
                    )
                if answers_llm_unedited is None or answer_llm_unedited == '':
                    raise Exception("No answer from LLM.")
                answers_llm_unedited.append(answer_llm_unedited)
                # if i < n_samples - 1:
                #     example = f"{human_prompt}\nCorrect answer (desired output): ```{ground_truth_answer}```"
                #     mutated_task_prompt = self.controller_mutation.mutate_current_prompt(
                #         n_mutations=1, prompt_for_mutation=system_prompt.split('\n\n')[0], output_example=example,
                #         use_my_mut_think=True
                #     )
                #     if mutated_task_prompt == "":
                #         mutated_task_prompt = task_prompts[0]
                #     task_prompts.append(mutated_task_prompt)
                #     system_prompt = f"{mutated_task_prompt}\n\n{system_prompts_output[answer_type]}\n\n{system_prompts_static[answer_type]}"
                if output_format != OutputFormat.NO_FORMAT:
                    processed_answer = final_answer
                    if reward_method == RewardMethod.MAJOR:
                        answer_for_voting.append(final_answer)
                    else:
                        answer_for_voting.append(cot_part)
                else:
                    processed_answer = ResultUtils.preprocess_answer(answer_llm_unedited, answer_type)
                    answer_for_voting.append(processed_answer)
                final_answers.append(processed_answer)
            answers_before_processing = '\n------\n'.join(answers_llm_unedited)
            answer_results.llm_answer_unedited = answers_before_processing
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                question_eval = ''.join([human_prompt.split('```')[1], human_prompt.split('```')[3]]).strip()
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
            score_answer = None
            if reward_method == RewardMethod.MAJOR:
                answer_llm_chosen = RewardMethods.majority_element(answer_for_voting)
            elif reward_method == RewardMethod.RERANK:
                chosen_answer_obj = self.reward_methods.get_reranking_model_best_answer(
                    question=question_eval, answer_options=answer_for_voting
                )
                answer_llm_chosen = chosen_answer_obj.chosen_answer
                score_answer = chosen_answer_obj.answer_score
            elif reward_method == RewardMethod.REWARD_M:
                chosen_answer_obj = self.reward_methods.get_reward_model_best_answer(
                    question=question_eval, answer_options=answer_for_voting
                )
                answer_llm_chosen = chosen_answer_obj.chosen_answer
                score_answer = chosen_answer_obj.answer_score
            elif reward_method == RewardMethod.LLM_O_B_I:
                chosen_answer_obj = self.reward_methods.get_llm_best_answer_best_idx(
                    question=question_eval, answer_options=answer_for_voting
                )
                answer_llm_chosen = chosen_answer_obj.chosen_answer
                score_answer = chosen_answer_obj.answer_score
            elif reward_method == RewardMethod.LLM_O_R:
                chosen_answer_obj = self.reward_methods.get_llm_best_answer_reranker(
                    question=question_eval, answer_options=answer_for_voting
                )
                answer_llm_chosen = chosen_answer_obj.chosen_answer
                score_answer = chosen_answer_obj.answer_score
            else:
                raise Exception(f"Reward method {reward_method.value} not implemented.")
            final_answer_chosen = None
            if answer_llm_chosen is not None:
                idx_chosen = answer_for_voting.index(answer_llm_chosen)
                final_answer_chosen = final_answers[idx_chosen]
                full_answer_chosen = answers_llm_unedited[idx_chosen]
                prompt_chosen = task_prompts[idx_chosen]

            answer_results.chosen_answer = full_answer_chosen if answer_llm_chosen is not None else None
            answer_results.task_prompts_all = '\n------\n'.join(task_prompts)
            answer_results.task_prompts_chosen = [prompt_chosen] if answer_llm_chosen is not None else []
            # answer_results.task_prompts_correct = '\n------\n'.join(correct_answer_task_prompts)
            answer_results.score_chosen = score_answer
            if final_answer_chosen is not None:
                correct = ResultUtils.check_corrct_answer(
                    llm_answer=final_answer_chosen, true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                    answer_type=answer_type
                )
            else:
                correct = False
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def get_answer_with_adaptive_mutation(
            self,
            system_prompt: str,
            human_prompt: str,
            model_name: str | None,
            answer_type: AnswerType,
            ground_truth_answer: str,
            result_file: str | None,
            ground_truth_answer_word: str | None = None,
            temperature: float = 0.0,
            output_format: OutputFormat = OutputFormat.STRUCTURED_COT,
            use_example_mut: bool = False,
            use_system_prompt_structure: bool = False,
            get_mut_scores: bool = False

    ) -> AnswerResults:
        """
            Gets answer with adaptive dynamic mutation - mutate only if answer was not correct. Mutations and their
            scores are saved.
        Args:
            system_prompt: System prompt. (Instructions)
            human_prompt: Human prompt. (Question and answer options)
            temperature: Temperature (0-2).
            model_name: Name of used LLM.
            answer_type: Type of answer. Enum from AnswerType.
            ground_truth_answer: Ground truth answer.
            result_file: File where the results are saved.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).
            output_format: Answer format time. OutputFormat.
            use_example_mut: If True, when mutating task prompt, gives an example for question and desired answer.
            use_system_prompt_structure: Whether to use system prompt structure or just plain system prompt.
            get_mut_scores: If True, gets reranking and reward model scores for mutated task prompt and question or
                            answer.
        Returns:
            AnswerResults, which contains the LLM output and if the answer was correct or not.

        """
        answer_results = AnswerResults()
        try:
            task_prompts = [system_prompt.split('\n\n')[0]]
            task_prompts_chosen = []
            start_mutation_idx = self.last_mutation_prompt_idx
            start_thinking_style_idx = self.last_thinking_style_idx
            extracted_answer = self.get_answer(
                output_format=output_format, system_prompt=system_prompt, human_prompt=human_prompt,
                temperature=temperature, model_name=model_name, answer_type=answer_type,
                ground_truth_answer=ground_truth_answer, ground_truth_answer_word=ground_truth_answer_word
            )
            answer_llm_unedited = extracted_answer.answer_llm_unedited
            processed_answer = extracted_answer.processed_answer
            correct = extracted_answer.correct
            # ja nav pareiza, mutē, kamēr iziet cauri visiem indexiem
            if not correct:
                all_combinations_reached = False
                if use_example_mut:
                    example = f"{human_prompt}\nCorrect answer (desired output): ```{ground_truth_answer}```"
                else:
                    example = None
                while not correct and not all_combinations_reached:
                # indexi kaut kā jāsakārto
                    start_task_prompt = system_prompt.split('\n\n')[0]
                    mutated_task_prompt = self.controller_mutation.mutate_current_prompt(
                        n_mutations=1, prompt_for_mutation=start_task_prompt, output_example=example,
                        use_my_mut_think=True, mutation_prompt_idx=self.last_mutation_prompt_idx,
                        thinking_style_idx=self.last_thinking_style_idx
                    )
                    used_mutation_prompt = my_mutation_prompts[self.last_mutation_prompt_idx]
                    used_thinking_style = my_thinking_styles[self.last_thinking_style_idx]
                    self.last_mutation_prompt_idx += 1
                    if self.last_mutation_prompt_idx > self.max_mutation_prompt_idx:
                        self.last_mutation_prompt_idx = 0
                    if self.last_mutation_prompt_idx == start_mutation_idx:
                        self.last_thinking_style_idx += 1
                        if self.last_thinking_style_idx > self.max_thinking_style_idx:
                            self.last_thinking_style_idx = 0
                        if self.last_thinking_style_idx == start_thinking_style_idx:
                            all_combinations_reached = True
                    task_prompts.append(mutated_task_prompt)
                    if use_system_prompt_structure:
                        system_prompt = (f"{mutated_task_prompt}\n\n{system_prompts_output[answer_type.value]}"
                                         f"\n\n{system_prompts_static[answer_type.value]}")
                    else:
                        system_prompt = mutated_task_prompt

                    extracted_answer = self.get_answer(
                        output_format=output_format, system_prompt=system_prompt, human_prompt=human_prompt,
                        temperature=1.0, model_name=model_name, answer_type=answer_type,
                        ground_truth_answer=ground_truth_answer, ground_truth_answer_word=ground_truth_answer_word
                    )
                    answer_llm_unedited = extracted_answer.answer_llm_unedited
                    processed_answer = extracted_answer.processed_answer
                    correct = extracted_answer.correct
                    # TODO error handling
                    reward_model_score_task_question = None
                    reranking_score_task_question = None
                    reward_model_score_answer_question = None
                    reranking_score_answer_question = None
                    question_raw = re.findall(r'\n```\n(.*?)\n```\n', human_prompt, re.DOTALL)[0]
                    if get_mut_scores:
                        reward_model_score_task_question = self.reward_methods.get_reward_model_score(
                            answer_option=mutated_task_prompt, question=question_raw
                        )
                        reranking_score_task_question = self.reward_methods.get_reranking_model_score(
                            answer_option=mutated_task_prompt, question=question_raw
                        )
                        if answer_llm_unedited is not None:
                            reranking_score_answer_question = self.reward_methods.get_reranking_model_score(
                                answer_option=answer_llm_unedited, question=question_raw
                            )
                            reward_model_score_answer_question = self.reward_methods.get_reward_model_score(
                                answer_option=answer_llm_unedited, question=question_raw
                            )
                        else:
                            reranking_score_answer_question = None
                            reward_model_score_answer_question = None

                    mutation_file = DynamicMutationInfo(
                        id=0,
                        start_task_prompt=start_task_prompt,
                        mutated_task_prompt=mutated_task_prompt,
                        llm_answer=answer_llm_unedited,
                        correct_answer=correct,
                        reward_model_name=self.reward_methods.reward_name,
                        reward_model_score_task_question=reward_model_score_task_question,
                        reranking_score_task_question=reranking_score_task_question,
                        reward_model_score_answer_question=reward_model_score_answer_question,
                        reranking_score_answer_question=reranking_score_answer_question,
                        mutation_prompt=used_mutation_prompt,
                        thinking_style=used_thinking_style,
                        question=question_raw,
                        llm_temperature=1.0,
                    )
                    self.save_mutation_file(mutation_info=mutation_file, result_file=result_file)
                if correct:
                    task_prompts_chosen.append(system_prompt.split('\n\n')[0])
            else:
                task_prompts_chosen.append(system_prompt.split('\n\n')[0])
            answer_results.llm_answer_unedited = answer_llm_unedited
            answer_results.chosen_answer = processed_answer
            answer_results.task_prompts_all = '\n------\n'.join(task_prompts)
            answer_results.task_system_prompts = task_prompts_chosen
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def save_mutation_file(self, mutation_info: DynamicMutationInfo, result_file: str) -> None:
        """
        Save the mutation info to the file.
        Args:
            mutation_info: Mutation info to save.
            result_file: File where the results are saved.
        """
        mutation_info.result_file = result_file.split('/')[-1]
        dataset = result_file.split('/')[result_file.split('/').index('datasets')+1]  # TODO
        saving_path_dataset = f'../datasets/{dataset}/results_mutation/info_mutation.csv'
        dataset_dir = os.path.dirname(saving_path_dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if not os.path.exists(saving_path_dataset):
            fieldnames = [field.name for field in DynamicMutationInfo.__dataclass_fields__.values()]
            with open(saving_path_dataset, 'w', newline='', encoding='utf-8') as resultsfile:
                writer = csv.DictWriter(resultsfile, fieldnames=fieldnames)
                writer.writeheader()
        mutation_info.id = FileUtils.get_highest_id_from_csv(saving_path_dataset) + 1
        data_to_append = asdict(mutation_info)
        with open(saving_path_dataset, 'a', newline='', encoding='utf-8') as csvfile:
            writer_info = csv.DictWriter(csvfile, fieldnames=data_to_append.keys())
            writer_info.writerow(data_to_append)

    def get_answer(
            self,
            output_format: OutputFormat,
            system_prompt: str,
            human_prompt: str,
            temperature: float,
            model_name: str | None,
            answer_type: AnswerType,
            ground_truth_answer: str,
            ground_truth_answer_word: str | None = None
    ) -> AnswerExtraction:
        """
        Get the answer from llm and check if it is correct or not.
        Args:
            output_format: Answer output format.
            system_prompt: System prompt.
            human_prompt: Human prompt.
            temperature: Temperature (0-2).
            model_name: Name of used LLM.
            answer_type: Type of answer. Enum from AnswerType.
            ground_truth_answer: Ground truth answer.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).

        Returns:
            AnswerExtraction, which contains the LLM output and if the answer was correct or not.

        """
        answer_extraction = AnswerExtraction()
        try:
            if output_format != OutputFormat.NO_FORMAT:
                answer_llm_structure = self.controller_ai.get_structured_output(
                    system_prompt=system_prompt, human_prompt=human_prompt, response_count=1,
                    temperature=temperature, model_name=model_name, answer_type=answer_type,
                    output_format=output_format
                )
                if answer_type == AnswerType.MULTIPLE_CHOICE:
                    final_answer = answer_llm_structure[0].answer_as_letter
                    final_answer_str = f"ANSWER_AS_LETTER: {final_answer}"
                elif answer_type == AnswerType.NUMBER:
                    final_answer = answer_llm_structure[0].answer_as_number
                    final_answer_str = f"ANSWER_AS_NUMBER: {final_answer}"
                else:
                    raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
                cot_part = answer_llm_structure[0].solution_explanation
                answer_llm_unedited = f"SOLUTION_EXPLANATION: {cot_part}\n{final_answer_str}"
            else:
                answer_llm_unedited = self.controller_ai.get_llm_api_response_with_backup_special(
                    system_prompt=system_prompt, prompt=human_prompt, response_count=1, temperature=temperature,
                    get_multiple_answers=False, model_name=model_name
                )
            if answer_llm_unedited is None or answer_llm_unedited == '':
                raise Exception("No answer from LLM.")
            answer_extraction.answer_llm_unedited = answer_llm_unedited
            if output_format != OutputFormat.NO_FORMAT:
                processed_answer = final_answer
            else:
                processed_answer = ResultUtils.preprocess_answer(answer_llm_unedited, answer_type)
            answer_extraction.processed_answer = processed_answer
            if processed_answer is not None:
                correct = ResultUtils.check_corrct_answer(
                    llm_answer=processed_answer, true_answer=ground_truth_answer,
                    other_true_answer=ground_truth_answer_word,
                    answer_type=answer_type
                )
            else:
                correct = False
            answer_extraction.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_extraction

    def get_structured_output(
            self,
            human_prompt: str,
            response_count: int,
            temperature: float,
            answer_type: AnswerType,
            ground_truth_answer: str,
            system_prompt: str | None = None,
            model_name: str | None = None,
            ground_truth_answer_word: str | None = None,
            output_format: OutputFormat = OutputFormat.STRUCTURED_COT

    ) -> AnswerResults:
        """
        Generate a structured output from the LLM and check if the answer is correct.

        Args:
            human_prompt: The input prompt/question for the LLM.
            response_count: Number of responses to generate.
            temperature: Sampling temperature for the LLM (0-2).
            answer_type: Type of the answer. AnswerType Enum.
            ground_truth_answer: The correct answer for validation.
            system_prompt: Optional system-level instructions for the LLM.
            model_name: Name of the LLM model to use.
            ground_truth_answer_word: Optional additional correct answer for validation.
            output_format: Format of the LLM's output .

        Returns:
            An object containing the LLM's output and whether the answer is correct.

        Raises:
            Exception: If the answer type is not implemented or if the answer format is invalid.
        """
        answer_results = AnswerResults()
        try:
            structured_answer = self.get_structured_output_llm(
                human_prompt=human_prompt, answer_type=answer_type, system_prompt=system_prompt, model_name=model_name,
                response_count=response_count, temperature=temperature, output_format=output_format
            )
            cot_part = structured_answer.solution_explanation
            extra_str = ''
            if output_format == OutputFormat.STRUCTURED_EXTRA:
                steps = '\n'.join(structured_answer.steps_for_answer)
                variables = '\n'.join(structured_answer.extracted_variables)
                extra_str = f"EXTRACTED_VARIABLES:\n{variables}\nSTEPS_FOR_ANSWER:\n{steps}"
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                final_answer = structured_answer.answer_as_letter
                final_answer_str = f"ANSWER_AS_LETTER: {final_answer}"
                if len(final_answer) != 1 or not final_answer.isupper():
                    raise Exception(f"Answer as letter should be one uppercase letter.\n COT part: {cot_part}\n"
                                    f"Answer as letter: {final_answer}")
            elif answer_type == AnswerType.NUMBER:
                final_answer = structured_answer.answer_as_number
                if output_format != OutputFormat.STRUCTURED_ANSWER and cot_part == '':
                    final_answer = ''
                final_answer_str = f"ANSWER_AS_NUMBER: {final_answer}"
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
            if output_format == OutputFormat.STRUCTURED_ANSWER:
                answer_results.llm_answer_unedited = final_answer
            else:
                answer_results.llm_answer_unedited = f"SOLUTION_EXPLANATION: {cot_part}\n{extra_str}\n{final_answer_str}"

            correct = ResultUtils.check_corrct_answer(
                llm_answer=str(final_answer), true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                answer_type=answer_type
            )
            answer_results.correct = correct

        except Exception as e:
            logger.error(e)
        return answer_results

    def get_structured_output_llm(
            self,
            human_prompt: str,
            answer_type: AnswerType,
            system_prompt: str | None = None,
            model_name: str | None = None,
            response_count: int = 1,
            temperature: float = 0.0,
            output_format: OutputFormat = OutputFormat.STRUCTURED_COT

    ) -> StructuredOutput:
        """
        Retrieve a structured output from the LLM based on the provided parameters.

        Args:
            human_prompt: The input prompt/question for the LLM.
            answer_type: The type of the answer. Enum AnswerType.
            system_prompt: Optional system-level instructions for the LLM.
            model_name: The name of the LLM model to use.
            response_count: The number of responses to generate. Defaults to 1.
            temperature: Sampling temperature for the LLM (0-2). Defaults to 0.0.
            output_format: The format of the LLM's output.

        Returns:
            A structured output object containing the LLM's response.

        Raises:
            Exception: If more than one structured output is returned by the LLM.

        """
        structured_answer = StructuredOutput()
        try:
            structured_answers_list = []
            structured_answers_list = self.controller_ai.get_structured_output(
                human_prompt=human_prompt, system_prompt=system_prompt, model_name=model_name,
                response_count=response_count, temperature=temperature, output_format=output_format,
                answer_type=answer_type
            )
            if len(structured_answers_list) > 1:
                raise Exception("Structured output should return only one answer.")
            elif len(structured_answers_list) == 1:
                structured_answer = structured_answers_list[0]
        except Exception as e:
            logger.error(e)
        return structured_answer

    def get_two_prompts_output(
            self,
            question_text: str,
            temperature: float,
            answer_type: AnswerType,
            ground_truth_answer: str,
            method: Method,
            ground_truth_answer_word: str | None = None,
            model_name: str | None = None,
            system_prompt: str | None = None,
            task_prompt: str | None = None

    ) -> AnswerResults:
        """
        Generate an answer using a two-prompt approach and validate its correctness.

        Args:
            question_text: The input question or problem statement.
            temperature: Sampling temperature for the LLM (0-2).
            answer_type: The type of the answer. Enum AnswerType.
            ground_truth_answer: The correct answer for validation.
            method: The method to use for generating the answer (e.g., PS, PS_PLUS, ZS_COT, TWO_PROMPTS).
            ground_truth_answer_word: Optional additional correct answer for validation.
            model_name: The name of the LLM model to use.
            system_prompt: Optional system-level instructions for the LLM.
            task_prompt: Optional task-specific instructions for the LLM.

        Returns:
            AnswerResults: An object containing the LLM's output and whether the answer is correct.

        Raises:
            Exception: If the specified method or answer type is not implemented.
        """
        answer_results = AnswerResults()
        try:
            if method == Method.PS:
                plan_str_answer = 'PLAN'
                plan_prompt = 'Let’s first understand the problem and devise a plan to solve the problem. ' \
                              'Then, let’s carry out the plan and solve the problem step by step.'
            elif method == Method.PS_PLUS:
                plan_str_answer = 'PLAN'
                plan_prompt = 'Let’s first understand the problem, extract relevant variables and their ' \
                              'corresponding numerals, and make a complete plan.Then, let’s carry out the plan, ' \
                              'calculate intermediate variables (pay attention to correct numerical calculation and ' \
                              'commonsense), solve the problem step by step, and show the answer.'
            elif method == Method.ZS_COT:
                plan_str_answer = 'COT'
                plan_prompt = "Let's think step by step."
            elif method == Method.TWO_PROMPTS:
                plan_str_answer = 'ANSWER_1'
                plan_prompt = task_prompt if task_prompt else ""

            else:
                raise Exception(f"Method {method.value} not yet implemented.")
            plan_prompt_full = f'{question_text}\n\nA: {plan_prompt}'
            if method == Method.TWO_PROMPTS and task_prompt is None:
                plan_prompt_full = question_text
            answer_llm_plan = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=plan_prompt_full, response_count=1, temperature=temperature,
                get_multiple_answers=False, model_name=model_name
            )[0]
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                answer_final_str = 'ANSWER_AS_LETTER'
                answer_prompt = 'Therefore, among A through D, the answer is' # TODO
            elif answer_type == AnswerType.NUMBER:
                answer_final_str = 'ANSWER_AS_NUMBER'
                answer_prompt = 'The answer (arabic numerals) is'
            elif answer_type == AnswerType.TEXT:
                answer_final_str = 'ANSWER_AS_TEXT'
                answer_prompt = 'The answer is'
            elif answer_type == AnswerType.BOOL:
                answer_final_str = 'ANSWER_AS_BOOL'
                answer_prompt = 'The answer (Yes or No) is'
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type.value}")
            final_answer_prompt_full = f"{plan_prompt_full}\n{answer_llm_plan}\n{answer_prompt}"
            if method == Method.ZS_COT:
                final_answer_prompt_full = f"{question_text}\nA: {answer_llm_plan}\n{answer_prompt}"
            final_answer = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=final_answer_prompt_full, response_count=1, temperature=temperature,
                get_multiple_answers=False, model_name=model_name
            )[0]

            answer_results.llm_answer_unedited = f"{plan_str_answer}: {answer_llm_plan}\n" \
                                                 f"{answer_final_str}: {final_answer}"
            answer_llm = ResultUtils.preprocess_answer(final_answer, answer_type)
            correct = ResultUtils.check_corrct_answer(
                llm_answer=answer_llm, true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                answer_type=answer_type
            )
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results
