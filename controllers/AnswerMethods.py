from typing import List
from loguru import logger
from controllers.AiLLM import ControllerAiLLM
from controllers.RewardMethods import RewardMethods
from controllers.Mutation import Mutation
from utils.result_utils import ResultUtils
from models.constants import system_prompts_output, system_prompts_static
from models.DataClass.AnswerResults import AnswerResults


class AnswerMethods:
    def __init__(self):
        self.controller_ai = ControllerAiLLM()
        self.controller_mutation = Mutation(controller_ai=self.controller_ai)

    def get_zero_shot_answer(
            self,
            system_prompt: str,
            human_prompt: str,
            temperature: float,
            model_name: str | None,
            answer_type: str,
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
            )
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
            answer_type: str,
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
            ground_truth_answer: Ground truth answer.
            ground_truth_answer_word: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).

        Returns:
            AnswerResults, which contains the LLM output and if the answer was correct or not.
        """
        answer_results = AnswerResults()
        try:
            answer_llm_unedited = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=human_prompt, response_count=response_count, temperature=temperature,
                get_multiple_answers=True, model_name=model_name
            )
            answers_before_processing = '\n------\n'.join(answer_llm_unedited)
            answer_results.llm_answer_unedited = answers_before_processing
            for answer_idx, answer_generated in enumerate(answer_llm_unedited):
                answer_llm_unedited[answer_idx] = ResultUtils.preprocess_answer(answer_generated, answer_type)
            answer_llm = RewardMethods.majority_element(answer_llm_unedited)
            answer_results.chosen_answer = answer_llm
            if answer_llm is not None:
                correct = ResultUtils.check_corrct_answer(
                    llm_answer=answer_llm, true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                    answer_type=answer_type
                )
            else:
                correct = False
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def get_answer_with_mutation(
            self,
            system_prompt: str,
            human_prompt: str,
            n_samples: int,
            temperature: float,
            model_name: str | None,
            answer_type: str,
            ground_truth_answer: str,
            ground_truth_answer_word: str | None = None
    ) -> AnswerResults:
        """

        Args:
            system_prompt: System prompt.
            human_prompt: Human prompt.
            n_samples: Count of iterated mutations.
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
            answers_llm_unedited = []
            processed_answers = []
            task_prompts = [system_prompt.split('\n\n')[0]]
            majority_task_prompts = []
            for i in range(n_samples):
                answer_llm_unedited = self.controller_ai.get_llm_api_response_with_backup_special(
                    system_prompt=system_prompt, prompt=human_prompt, response_count=1, temperature=temperature,
                    get_multiple_answers=False, model_name=model_name
                )
                if answers_llm_unedited is None or answer_llm_unedited == '':
                    raise Exception("No answer from LLM.")
                answers_llm_unedited.append(answer_llm_unedited)
                if i < n_samples - 1:
                    example = f"{human_prompt}\nCorrect answer (desired output): ```{ground_truth_answer}```"
                    mutated_task_prompt = self.controller_mutation.mutate_current_prompt(
                        n_mutations=1, prompt_for_mutation=system_prompt.split('\n\n')[0], output_example=example
                    )
                    task_prompts.append(mutated_task_prompt)
                    system_prompt = f"{mutated_task_prompt}\n\n{system_prompts_output[answer_type]}\n\n{system_prompts_static[answer_type]}"
                processed_answer = ResultUtils.preprocess_answer(answer_llm_unedited, answer_type)
                processed_answers.append(processed_answer)
            answers_before_processing = '\n------\n'.join(answers_llm_unedited)
            answer_results.llm_answer_unedited = answers_before_processing
            answer_llm = RewardMethods.majority_element(processed_answers)
            if answer_llm is not None:
                for idx in range(n_samples):
                    if processed_answers[idx] == answer_llm:
                        majority_task_prompts.append(task_prompts[idx])

            answer_results.chosen_answer = answer_llm
            answer_results.task_prompts_all = '\n------\n'.join(task_prompts)
            answer_results.task_prompts_majority = '\n------\n'.join(majority_task_prompts)
            if answer_llm is not None:
                correct = ResultUtils.check_corrct_answer(
                    llm_answer=answer_llm, true_answer=ground_truth_answer, other_true_answer=ground_truth_answer_word,
                    answer_type=answer_type
                )
            else:
                correct = False
            answer_results.correct = correct
        except Exception as e:
            logger.error(e)
        return answer_results

    def get_structured_output(self):
        pass


