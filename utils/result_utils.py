import csv
from models.Enums.AnswerType import AnswerType
from loguru import logger
import re


class ResultUtils:

    @staticmethod
    def count_correct_values(file_path: str, output_result: bool = False) -> float:
        try:
            total_values = 0
            correct_values = 0

            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    total_values += 1
                    if row['correct'] == 'True':
                        correct_values += 1

            result = correct_values / total_values if total_values > 0 else 0
            if output_result:
                print(f'{correct_values}/{total_values} = {result}')
        except Exception as e:
            print(f'Error: {e}')
            result = 0.0
        return result

    @staticmethod
    def check_corrct_answer(
            llm_answer: str,
            true_answer: str,
            answer_type: str,
            other_true_answer: str | None = None
    ) -> bool:
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
        preprocessed_answer = llm_answer
        try:
            if '\n' in preprocessed_answer or len(preprocessed_answer) > 10:
                preprocessed_answer = llm_answer.split('\n')[-1]
            preprocessed_answer = preprocessed_answer.lower()
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
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
                                found_letters = re.findall(r'(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])', preprocessed_answer)
                                if len(found_letters) == 1:
                                    preprocessed_answer = found_letters[0]
                                else:
                                    valid_found_letters = [letter for letter in found_letters if letter in ['a', 'b', 'c', 'd', 'e']]
                                    if len(valid_found_letters) == 1:
                                        preprocessed_answer = valid_found_letters[0]
                                    else:
                                        preprocessed_answer = valid_found_letters[-1]
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
                            a=0

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
        return preprocessed_answer


if __name__ == '__main__':
    path = '../datasets/not_using/StrategyQA/results/a1_11-02-2025.csv'
    path_2 = '../datasets/not_using/StrategyQA_2/results/a1_11-02-2025.csv'
    ResultUtils.count_correct_values(path, True)
    ResultUtils.count_correct_values(path_2, True)
