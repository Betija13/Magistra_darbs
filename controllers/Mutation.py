import random
import json
from typing import List, Tuple
from loguru import logger
from collections import Counter
from models.constants import mutation_prompts, thinking_styles, problem_descriptions, my_mutation_prompts, my_thinking_styles
from models.DataClass.LLMPrompt import LLMPrompt
from controllers.AiLLM import ControllerAiLLM


class Mutation:
    def __init__(self, controller_ai: ControllerAiLLM | None = None):
        if controller_ai is not None:
            self.controller_ai = controller_ai
        else:
            self.controller_ai = ControllerAiLLM()
        self.mutation_prompt_index: int
        self.thinking_style_index: int
        try:
            with open('../models/other/good_mutation_combinations.json', 'r') as file:
                self.good_mutation_combinations: List[List[int]] = json.load(file)
        except FileNotFoundError:
            self.good_mutation_combinations: List[List[int]] = [[4, 19], [6, 13], [6, 0], [4, 16], [37, 39]]
        try:
            with open('../models/other/bad_mutation_combinations.json', 'r') as file:
                self.bad_mutation_combinations: List[List[int]] = json.load(file)
        except FileNotFoundError:
            self.bad_mutation_combinations: List[List[int]] = [[12, 44], [8, 46], [13, 4], [14, 25], [27, 33], [7, 49], [21, 39], [37, 53], [5, 51], [1, 36]]
        try:
            with open('../models/other/bad_thinking_style_indexes.json', 'r') as file:
                self.bad_thinking_style_indexes: List[int] = json.load(file)
        except FileNotFoundError:
            self.bad_thinking_style_indexes: List[int] = []
        try:
            with open('../models/other/bad_mutation_indexes.json', 'r') as file:
                self.bad_mutation_indexes: List[int] = json.load(file)
        except FileNotFoundError:
            self.bad_mutation_indexes: List[int] = [25, 49]

    def save_lists_combination(self, file_name: str, object_saved: List[List[int]]):
        with open(file_name, 'w') as file:
            json.dump(object_saved, file)

    def save_lists(self, file_name: str, object_saved: List[int]):
        with open(file_name, 'w') as file:
            json.dump(object_saved, file)

    def mutate_source_prompt(
            self,
            n_samples: int = 3,
            prompt_for_mutation: str | None = None,
            output_results: bool = False,
            user_feedback: bool = False,
            output_example: str | None = None,
            use_my_mut_think: bool = True
    ) -> List[str]:
        """
        Mutate the source prompt. Output is n_samples of mutated prompts.
        Args:
            n_samples: Desired output count.
            prompt_for_mutation: Original prompt that needs to be mutated.
            output_results: Whether to output the results.
            user_feedback: Whether to ask for user feedback about chosen prompts.
            output_example: Example of the problem and answer.
            use_my_mut_think: Boolean about whether to use my created thinking and mutation styles.
        Returns:
            List of mutated prompts.
        """
        if use_my_mut_think:
            user_feedback = False

        prompt_for_mutation = self.get_pb_mutation_prompt(
            used_problem_description=prompt_for_mutation, output_prompt=output_results, output_example=output_example,
            use_my_mut_think=use_my_mut_think
        )
        human_prompt_for_mutation = prompt_for_mutation.human_prompt
        system_prompt_for_mutation = prompt_for_mutation.system_prompt
        list_of_mutations = self.controller_ai.get_llm_api_response_with_backup_special(
            prompt=human_prompt_for_mutation, system_prompt=system_prompt_for_mutation, response_count=n_samples,
            temperature=1.0, model_name='gpt-4o', get_multiple_answers=True
        )
        for idx, mutation_n in enumerate(list_of_mutations):
            mutation_n = mutation_n.replace('INSTRUCTION MUTANT:', '').strip()
            list_of_mutations[idx] = mutation_n
        if output_results:
            for result in list_of_mutations:
                logger.info(result)
                logger.info("-------------------")
        if user_feedback:
            good_mutation = input('Good result? [y/n]')
            indexes_t_m = [self.thinking_style_index, self.mutation_prompt_index]
            if good_mutation == 'n':
                self.bad_mutation_combinations.append(indexes_t_m)
                self.save_lists_combination(file_name='../models/other/bad_mutation_combinations.json',
                                            object_saved=self.bad_mutation_combinations)
            elif good_mutation == 'y':
                self.good_mutation_combinations.append(indexes_t_m)
                self.save_lists_combination(file_name='../models/other/good_mutation_combinations.json',
                                            object_saved=self.good_mutation_combinations)
            self.calculate_add_bad_examples()
        return list_of_mutations

    def mutate_current_prompt(
            self,
            n_mutations: int,
            prompt_for_mutation: str | None = None,
            output_results: bool = False,
            user_feedback: bool = False,
            output_example: str | None = None,
            use_my_mut_think: bool = False
    ) -> str:
        """
        Mutate the current prompt n_mutation times. Output is the last mutated prompt.
        Args:
            n_mutations: Count of mutations.
            prompt_for_mutation: Original prompt that needs to be mutated.
            output_results: Whether to output the results.
            user_feedback: Whether to ask for user feedback about chosen prompts.
            output_example: Example of the problem and answer.
            use_my_mut_think: Boolean about whether to use my created thinking and mutation styles.
        Returns:
            Last mutated prompt.
        """
        for _ in range(n_mutations):
            prompt_for_mutation = self.get_pb_mutation_prompt(
                used_problem_description=prompt_for_mutation, output_prompt=output_results,
                output_example=output_example, use_my_mut_think=use_my_mut_think
            )
            human_prompt_for_mutation = prompt_for_mutation.human_prompt
            system_prompt_for_mutation = prompt_for_mutation.system_prompt
            new_prompt = self.controller_ai.get_llm_api_response_with_backup_special(
                prompt=human_prompt_for_mutation, system_prompt=system_prompt_for_mutation, response_count=1,
                temperature=0.0, model_name='gpt-4o', get_multiple_answers=False
            )
            new_prompt = new_prompt.replace('INSTRUCTION MUTANT:', '').strip()
            if output_results:
                logger.info(new_prompt)
                logger.info("-------------------")
            if user_feedback:
                good_mutation = input('Good result? [y/n]')
                indexes_t_m = [self.thinking_style_index, self.mutation_prompt_index]
                if good_mutation == 'n':
                    self.bad_mutation_combinations.append(indexes_t_m)
                    self.save_lists_combination(file_name='../models/other/bad_mutation_combinations.json',
                                                object_saved=self.bad_mutation_combinations)
                elif good_mutation == 'y':
                    self.good_mutation_combinations.append(indexes_t_m)
                    self.save_lists_combination(file_name='../models/other/good_mutation_combinations.json',
                                                object_saved=self.good_mutation_combinations)
                    prompt_for_mutation = new_prompt
                self.calculate_add_bad_examples()
            else:
                prompt_for_mutation = new_prompt
        return prompt_for_mutation

    def get_pb_mutation_prompt(
            self, used_problem_description: str | None = None,
            output_prompt: bool = False,
            output_example: str | None = None,
            use_my_mut_think: bool = False
    ) -> LLMPrompt:
        """
        Get the structure for prompt, that will be given for mutator as instruction.
        Args:
            used_problem_description: Prompt that needs to be mutated. Chosen from pre-defined problem descriptions if
            None.
            output_prompt: Whether to output the prompt.
            output_example: Example of the problem and answer.
            use_my_mut_think: Boolean about whether to use my created thinking and mutation styles.
        Returns:
            Prompt for the mutator, that combines task prompt and mutation prompt.

        """
        self.thinking_style_index = -1
        self.mutation_prompt_index = -1
        if used_problem_description is None:
            used_problem_description = problem_descriptions['AQuA-RAT']
        if use_my_mut_think:
            used_thinking_style = random.choice(my_thinking_styles)
            used_mutation_prompt = random.choice(my_mutation_prompts)
        else:
            valid_thinking_style: bool = False
            while not valid_thinking_style:
                used_thinking_style = random.choice(thinking_styles)
                thinking_style_index = thinking_styles.index(used_thinking_style)
                if thinking_style_index not in self.bad_thinking_style_indexes:
                    valid_thinking_style = True
                    self.thinking_style_index = thinking_style_index
            valid_mutation_prompt: bool = False
            while not valid_mutation_prompt:
                used_mutation_prompt = random.choice(mutation_prompts)
                mutation_prompt_index = mutation_prompts.index(used_mutation_prompt)
                if mutation_prompt_index not in self.bad_mutation_indexes:
                    valid_mutation_prompt = True
                    self.mutation_prompt_index = mutation_prompt_index
        example_str = ""
        if output_example is not None:
            example_str = f"\n\n--\n\nFor context: The given INSTRUCTION MUTANT will be used to get an answer for a question. Question and answer examples: {output_example}"
        return_system_prompt = f"{used_thinking_style} {used_mutation_prompt} \n\nReturn only the created INSTRUCTION MUTANT." \
                        f"\n\nMake sure your answer is short, preferably one sentence. Do not give examples. Always answer with instruction mutant. Do not add any symbols, like ```, around it.{example_str}"
        return_human_prompt = f"INSTRUCTION: ```{used_problem_description}``` INSTRUCTION MUTANT:"
        return_prompt = LLMPrompt(system_prompt=return_system_prompt, human_prompt=return_human_prompt)
        if output_prompt:
            if not use_my_mut_think:
                logger.debug(f"Index of thinking style: {thinking_style_index}\tIndex of mutation prompt: {mutation_prompt_index}")
            logger.debug(return_prompt)
        return return_prompt

    def calculate_add_bad_examples(self):
        bad_thinking_styles, bad_mutation_prompts = zip(*self.bad_mutation_combinations)
        result_thinking_styles = dict(Counter(bad_thinking_styles))
        result_mutation_prompts = dict(Counter(bad_mutation_prompts))
        good_thinking_styles, good_mutation_prompts = zip(*self.good_mutation_combinations)
        result_good_thinking_styles = dict(Counter(good_thinking_styles))
        result_good_mutation_prompts = dict(Counter(good_mutation_prompts))
        for key, value in result_thinking_styles.items():
            if value > 2:
                if result_good_thinking_styles.get(key) and result_good_thinking_styles[key] <= 2:
                    if key not in self.bad_thinking_style_indexes:
                        self.bad_thinking_style_indexes.append(key)
                        self.save_lists(file_name='bad_thinking_style_indexes.json',
                                        object_saved=self.bad_thinking_style_indexes)
        for key, value in result_mutation_prompts.items():
            if value > 2:
                if result_good_mutation_prompts.get(key) and result_good_mutation_prompts[key] <= 2:
                    if key not in self.bad_mutation_indexes:
                        self.bad_mutation_indexes.append(key)
                        self.save_lists(file_name='bad_mutation_indexes.json',
                                        object_saved=self.bad_mutation_indexes)

    def create_initial_prompt(self, dataset: str):
        pass  # TODO


if __name__ == '__main__':
    mutation = Mutation()
    original_prompt = 'Solve the math word problem, giving your answer as an arabic numeral.'
    # full_prompt = '```{}\n{}\n{}```'
    for _ in range(5):
        final_prompts = mutation.mutate_source_prompt(
            prompt_for_mutation=original_prompt, n_samples=3, output_results=True, user_feedback=False,
            use_my_mut_think=True
        )
        logger.success('\n'.join(final_prompts))
        print('++++++++++++++')
    print('************')
    for _ in range(5):
        final_prompt = mutation.mutate_current_prompt(
            prompt_for_mutation=original_prompt, n_mutations=5, output_results=True, user_feedback=False,
            use_my_mut_think=True
        )
        logger.success(final_prompt)
