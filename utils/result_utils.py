import csv
import sys
from models.Enums.AnswerType import AnswerType
from models.DataClass.NumericResults import NumericResults
from loguru import logger
import re
from utils.OPRO_evaluation import OPROEvaluation


class ResultUtils:

    @staticmethod
    def count_correct_values(file_path: str, output_result: bool = False) -> NumericResults:
        """
        Count the number of correct values in the file, as well as count of short answers.
        Args:
            file_path: Path to the file.
            output_result: If the result should be printed.

        Returns:
            Percentage of correct values and percentage of short answers.
        """
        results = NumericResults(
            accuracy_score=0.0,
            percentage_of_short_answers=0.0
        )
        try:
            total_values = 0
            correct_values = 0
            count_of_small_results = 0
            # Increase the field size limit
            max_int = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(max_int)
                    break
                except OverflowError:
                    max_int = int(max_int / 10)

            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        total_values += 1
                        if row['correct'] == 'True':
                            correct_values += 1
                        llm_answer = row['llm_answer']
                        divider = '\n------\n'
                        if len(llm_answer.split(divider)) > 1:
                            small_res = 0
                            total_res = len(llm_answer.split(divider))
                            for answer_1 in llm_answer.split(divider):
                                if answer_1.count('\n') <= 0:
                                    small_res += 1
                            if small_res / total_res > 0.5:
                                count_of_small_results += 1
                        else:
                            if llm_answer.count('\n') <= 0:
                                count_of_small_results += 1
                    except Exception as e:
                        logger.error(e)

            results.accuracy_score = (correct_values / total_values) * 100 if total_values > 0 else None
            results.percentage_of_short_answers = (count_of_small_results / total_values) * 100 if \
                total_values > 0 else None
            if output_result:
                print(f'{correct_values}/{total_values} = {results.accuracy_score}')
        except Exception as e:
            logger.exception(e)
            print(f'Error: {e}')
        return results

    @staticmethod
    def check_corrct_answer(
            llm_answer: str,
            true_answer: str,
            answer_type: str,
            other_true_answer: str | None = None
    ) -> bool:
        """
        Check if the answer is correct.
        Args:
            llm_answer: Answer provided by the LLM.
            true_answer: Ground truth answer.
            answer_type: Answer type. Enum from AnswerType.
            other_true_answer: Word for ground truth answer (only if AnswerType.MULTIPLE_CHOICE.value).

        Returns:

        """
        answer_correct = False
        llm_answer = ResultUtils.preprocess_answer(llm_answer, answer_type)
        try:
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                if llm_answer.lower() == true_answer.lower() or \
                        (other_true_answer is not None and llm_answer.lower() == other_true_answer.lower()):
                    answer_correct = True
            elif answer_type == AnswerType.TEXT.value:
                if llm_answer.lower() == true_answer.lower():
                    answer_correct = True
            elif answer_type == AnswerType.NUMBER.value:
                if llm_answer.lower() == true_answer.lower():
                    answer_correct = True
                else:
                    try:
                        if float(llm_answer) == float(true_answer):
                            answer_correct = True
                        elif abs(float(llm_answer) - float(true_answer)) < float(true_answer)/100:
                            answer_correct = True
                    except Exception as e:
                        logger.error(e)
            elif answer_type == AnswerType.BOOL.value:
                answer_bool: bool | None = True if llm_answer.lower() == 'yes' else (
                    False if llm_answer.lower() == 'no' else None)
                if answer_bool is None:
                    pass
                if answer_bool is not None:
                    true_bool = true_answer == 'True'
                    answer_correct = answer_bool == true_bool
            else:
                raise Exception(f'Answer type {answer_type} not supported')
        except Exception as e:
            logger.error(e)
        return answer_correct

    @staticmethod
    def preprocess_answer(
            llm_answer: str,
            answer_type: str,
    ) -> str:
        """
        Preprocess the answer to match as much as possible to the desired answer.
        Args:
            llm_answer: Answer provided by the LLM.
            answer_type: answer_type: Answer type. Enum from AnswerType.

        Returns:
            Preprocessed answer.
        """
        preprocessed_answer = llm_answer
        try:
            regex_letters = r'(?<![a-zA-Z])[a-eA-E](?![a-zA-Z])'
            # regex_numbers = r'-?\d+\.\d+|-?\d+'
            lines = []
            answer_lines = []
            answers_in_boxed = []
            answers_in_stars = []
            if '\n' in preprocessed_answer or len(preprocessed_answer) > 10:
                lines = llm_answer.split('\n')
                answer_lines = [line for line in lines if 'answer' in line.lower() or 'correct choice' in line.lower()]
                answers_in_boxed = [ans for ans in re.compile(r'boxed\{(.*?)\}').findall(llm_answer)]
                # answers_in_stars = [ans for ans in re.compile(r'\*\*(.*?)\*\*').findall(llm_answer)]
            preprocessed_answer = preprocessed_answer.lower()
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                valid_lines = [line for line in lines if re.search(regex_letters, line)]
                valid_answer_lines = [line for line in answer_lines if re.search(regex_letters, line)]
                valid_boxed = [ans for ans in answers_in_boxed if re.search(regex_letters, ans)]
                valid_stars = [ans for ans in answers_in_stars if re.search(regex_letters, ans)]
                if len(valid_answer_lines) > 0:
                    last_line_with_answer_word = valid_answer_lines[-1]
                    if re.search(regex_letters, last_line_with_answer_word):
                        preprocessed_answer = last_line_with_answer_word
                    else:
                        possible_answer = valid_lines[valid_lines.index(last_line_with_answer_word) + 1]
                        if re.search(regex_letters, possible_answer):
                            preprocessed_answer = possible_answer
                if len(valid_boxed) > 0 and preprocessed_answer.count('\n') > 0:
                    preprocessed_answer = valid_boxed[-1]
                if len(valid_stars) > 0 and preprocessed_answer.count('\n') > 0:
                    preprocessed_answer = valid_stars[-1]
                if preprocessed_answer.count('\n') > 0:
                    if len(valid_lines) > 0:
                        preprocessed_answer = valid_lines[-1]
                if len(preprocessed_answer) > 1:
                    preprocessed_answer = preprocessed_answer.split('\n')[-1]
                    if len(preprocessed_answer) > 1:
                        if '(' in preprocessed_answer and ')' in preprocessed_answer:
                            answer_from_brackets = preprocessed_answer.split('(')[-1].split(')')[0].strip()
                            if len(answer_from_brackets) == 1:
                                preprocessed_answer = answer_from_brackets
                        if ')' in preprocessed_answer and preprocessed_answer[1] == ')':
                            preprocessed_answer = preprocessed_answer.split(')')[0].strip()
                        if len(preprocessed_answer) > 1:
                            if preprocessed_answer.count(')') == 1 and preprocessed_answer.split(')')[0][-2] == ' ':
                                preprocessed_answer = preprocessed_answer.split(')')[0][-1].strip()
                            else:
                                found_letters = re.findall(regex_letters, preprocessed_answer)
                                if len(found_letters) == 1:
                                    preprocessed_answer = found_letters[0]
                                else:
                                    valid_found_letters = [letter for letter in found_letters if letter.lower() in ['a', 'b', 'c', 'd', 'e']]
                                    if len(valid_found_letters) == 1:
                                        preprocessed_answer = valid_found_letters[0]
                                    else:
                                        preprocessed_answer = ""#valid_found_letters[-1]
            elif answer_type == AnswerType.TEXT.value:
                preprocessed_answer = llm_answer.replace(' ', '')
            elif answer_type == AnswerType.NUMBER.value:
                try:
                    answer_as_digits = float(preprocessed_answer)
                except Exception as e:
                    numbers_in_answer = re.findall(r'-?\d+\.\d+', preprocessed_answer)
                    if len(numbers_in_answer) > 1:
                        preprocessed_answer = numbers_in_answer[-1]
                    elif len(numbers_in_answer) == 1:
                        preprocessed_answer = numbers_in_answer[0]
                    else:
                        numbers_just_digits = re.findall(r'-?\d+', preprocessed_answer)
                        if len(numbers_just_digits) > 1:
                            preprocessed_answer = numbers_just_digits[-1]
                        elif len(numbers_just_digits) == 1:
                            preprocessed_answer = numbers_just_digits[0]
                        else:
                            preprocessed_answer = ''

            elif answer_type == AnswerType.BOOL.value:
                if 'yes' in preprocessed_answer.lower():
                    preprocessed_answer = 'yes'
                elif 'no' in preprocessed_answer.lower():
                    preprocessed_answer = 'no'
                else:
                    a=0
            else:
                raise Exception(f'Answer type {answer_type} not supported')
        except Exception as e:
            logger.error(e)
            preprocessed_answer = ''
        # opro_preprocess = OPROEvaluation.get_normalized_prediction(
        #     prediction=llm_answer, treat_as_number=answer_type==AnswerType.NUMBER.value, treat_as_bool=answer_type==AnswerType.BOOL.value)
        # if opro_preprocess != preprocessed_answer and preprocessed_answer not in opro_preprocess and opro_preprocess[0].lower() != preprocessed_answer.lower():
        #     a=0
        return preprocessed_answer

# ResultUtils.count_correct_values(file_path='../datasets/AQuA-RAT/results/A_ZERO_SHOT_03-04-2025_data_normalized_test_all_dataset_9.csv', output_result=True)

