from typing import List
from loguru import logger
from controllers.AiLLM import ControllerAiLLM
from controllers.RewardMethods import RewardMethods
from controllers.Mutation import Mutation
from utils.result_utils import ResultUtils
from models.constants import system_prompts_output, system_prompts_static
from models.DataClass.AnswerResults import AnswerResults
from models.DataClass.StructuredOutput import StructuredOutput
from models.Enums.AnswerType import AnswerType
from models.Enums.Method import Method

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
            ground_truth_answer_word: str | None = None,
            get_structured_output: bool = False # TODO maybe change this to OutputType Enum ? [structured_cot, structured_answer, regular]
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
            correct_answer_task_prompts = []
            for i in range(n_samples):
                if get_structured_output:
                    answer_llm_structure = self.controller_ai.get_structured_output_multiple_choice(
                        system_prompt=system_prompt, human_prompt=human_prompt, response_count=1,
                        temperature=temperature, model_name=model_name, answer_type=answer_type
                    )
                    if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                        final_answer = answer_llm_structure[0].answer_as_letter
                        final_answer_str = f"ANSWER_AS_LETTER: {final_answer}"
                    elif answer_type == AnswerType.NUMBER.value:
                        final_answer = answer_llm_structure[0].answer_as_number
                        final_answer_str = f"ANSWER_AS_NUMBER: {final_answer}"
                    else:
                        raise Exception(f"Answer extraction not yet implemented for {answer_type}")
                    cot_part = answer_llm_structure[0].solution_explanation
                    answer_llm_unedited = f"SOLUTION_EXPLANATION: {cot_part}\n{final_answer_str}"
                else:
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
                        n_mutations=1, prompt_for_mutation=system_prompt.split('\n\n')[0], output_example=example,
                        use_my_mut_think=True
                    )
                    task_prompts.append(mutated_task_prompt)
                    system_prompt = f"{mutated_task_prompt}\n\n{system_prompts_output[answer_type]}\n\n{system_prompts_static[answer_type]}"
                if get_structured_output:
                    processed_answer = final_answer
                else:
                    processed_answer = ResultUtils.preprocess_answer(answer_llm_unedited, answer_type)
                processed_answers.append(processed_answer)
            answers_before_processing = '\n------\n'.join(answers_llm_unedited)
            answer_results.llm_answer_unedited = answers_before_processing
            answer_llm = RewardMethods.majority_element(processed_answers)
            if answer_llm is not None:
                for idx in range(n_samples):
                    if processed_answers[idx] == answer_llm:
                        majority_task_prompts.append(task_prompts[idx])
                    if processed_answers[idx].lower() == ground_truth_answer.lower():
                        correct_answer_task_prompts.append(task_prompts[idx])

            answer_results.chosen_answer = answer_llm
            answer_results.task_prompts_all = '\n------\n'.join(task_prompts)
            answer_results.task_prompts_majority = '\n------\n'.join(majority_task_prompts)
            answer_results.task_prompts_correct = '\n------\n'.join(correct_answer_task_prompts)
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

    def get_structured_output(
            self,
            human_prompt: str,
            response_count: int,
            temperature: float,
            answer_type: str,
            ground_truth_answer: str,
            system_prompt: str | None = None,
            model_name: str | None = None,
            ground_truth_answer_word: str | None = None,
            only_answer: bool = False

    ) -> AnswerResults:
        answer_results = AnswerResults()
        try:
            structured_answer = self.get_structured_output_llm(
                human_prompt=human_prompt, answer_type=answer_type, system_prompt=system_prompt, model_name=model_name,
                response_count=response_count, temperature=temperature, only_answer=only_answer
            )
            cot_part = structured_answer.solution_explanation
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                final_answer = structured_answer.answer_as_letter
                final_answer_str = f"ANSWER_AS_LETTER: {final_answer}"
                if len(final_answer) != 1 or not final_answer.isupper():
                    raise Exception(f"Answer as letter should be one uppercase letter.\n COT part: {cot_part}\n"
                                    f"Answer as letter: {final_answer}")
            elif answer_type == AnswerType.NUMBER.value:
                final_answer = structured_answer.answer_as_number
                if not only_answer and cot_part == '':
                    final_answer = ''
                final_answer_str = f"ANSWER_AS_NUMBER: {final_answer}"
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type}")
            if only_answer:
                answer_results.llm_answer_unedited = final_answer
            else:
                answer_results.llm_answer_unedited = f"SOLUTION_EXPLANATION: {cot_part}\n{final_answer_str}"

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
            answer_type: str,
            system_prompt: str | None = None,
            model_name: str | None = None,
            response_count: int = 1,
            temperature: float = 0.0,
            only_answer: bool = False

    ) -> StructuredOutput:
        structured_answer = StructuredOutput()
        try:
            structured_answers_list = []
            structured_answers_list = self.controller_ai.get_structured_output_multiple_choice(
                human_prompt=human_prompt, system_prompt=system_prompt, model_name=model_name,
                response_count=response_count, temperature=temperature, only_answer=only_answer,
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
            answer_type: str,
            ground_truth_answer: str,
            method: str,
            ground_truth_answer_word: str | None = None,
            model_name: str | None = None,
            system_prompt: str | None = None,
            task_prompt: str | None = None

    ) -> AnswerResults:
        answer_results = AnswerResults()
        try:
            if method == Method.PS.value:
                plan_str_answer = 'PLAN'
                plan_prompt = 'Let’s first understand the problem and devise a plan to solve the problem. ' \
                              'Then, let’s carry out the plan and solve the problem step by step.'
            elif method == Method.PS_PLUS.value:
                plan_str_answer = 'PLAN'
                plan_prompt = 'Let’s first understand the problem, extract relevant variables and their ' \
                              'corresponding numerals, and make a complete plan.Then, let’s carry out the plan, ' \
                              'calculate intermediate variables (pay attention to correct numerical calculation and ' \
                              'commonsense), solve the problem step by step, and show the answer.'
            elif method == Method.ZS_COT.value:
                plan_str_answer = 'COT'
                plan_prompt = "Let's think step by step."
            elif method == Method.TWO_PROMPTS.value:
                plan_str_answer = 'ANSWER_1'
                plan_prompt = task_prompt if task_prompt else ""

            else:
                raise Exception(f"Method {method} not yet implemented.")
            plan_prompt_full = f'{question_text}\n\nA: {plan_prompt}'
            if method == Method.TWO_PROMPTS.value and task_prompt is None:
                plan_prompt_full = question_text
            answer_llm_plan = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=plan_prompt_full, response_count=1, temperature=temperature,
                get_multiple_answers=False, model_name=model_name
            )
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                answer_final_str = 'ANSWER_AS_LETTER'
                answer_prompt = 'Therefore, among A through E, the answer is'
            elif answer_type == AnswerType.NUMBER.value:
                answer_final_str = 'ANSWER_AS_NUMBER'
                answer_prompt = 'The answer (arabic numerals) is'
            elif answer_type == AnswerType.TEXT.value:
                answer_final_str = 'ANSWER_AS_TEXT'
                answer_prompt = 'The answer is'
            elif answer_type == AnswerType.BOOL.value:
                answer_final_str = 'ANSWER_AS_BOOL'
                answer_prompt = 'The answer (Yes or No) is'
            else:
                raise Exception(f"Answer extraction not yet implemented for {answer_type}")
            final_answer_prompt_full = f"{plan_prompt_full}\n{answer_llm_plan}\n{answer_prompt}"
            if method == Method.ZS_COT.value:
                final_answer_prompt_full = f"{question_text}\nA: {answer_llm_plan}\n{answer_prompt}"
            final_answer = self.controller_ai.get_llm_api_response_with_backup_special(
                system_prompt=system_prompt, prompt=final_answer_prompt_full, response_count=1, temperature=temperature,
                get_multiple_answers=False, model_name=model_name
            )

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


