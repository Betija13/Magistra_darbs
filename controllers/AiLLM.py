import os
import sys
from typing import List, Optional
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from openai import OpenAI
import os
from models.Enums.OutputFormat import OutputFormat
from models.DataClass.LLMModel import LLMModel
from models.DataClass.StructuredOutput import StructuredOutput, StructuredOutputModelMultipleChoice, \
    StructuredOutputModelMultipleChoiceOnlyChoice, StructuredOutputModelNumber, StructuredOutputModelNumberOnlyNumber, \
    StructuredOutputModelMultipleChoiceExtra
from models.Enums.AnswerType import AnswerType
from dotenv import dotenv_values
from loguru import logger

env_path = '.env'
if os.path.exists("../.env"):
    env_path = "../.env"
elif os.path.exists(".env"):
    env_path = ".env"
elif os.path.exists("../../.env"):
    env_path = "../../.env"
else:
    logger.critical("No .env file found")
config = dotenv_values(env_path)

# LLM_MAIN_MODEL = config.get('LLM_MAIN_MODEL')
LLM_OPENAI_API_KEY = config.get('LLM_OPENAI_API_KEY')
if LLM_OPENAI_API_KEY is None:
    logger.critical(f"LLM_OPENAI_API_KEY is None. Please add your OpenAi key as LLM_OPENAI_API_KEY to .env file")

class ControllerAiLLM:
    def __init__(self):
        self.model: LLMModel = LLMModel(
            name='gpt-4o', #LLM_MAIN_MODEL,
            api_key=LLM_OPENAI_API_KEY
        )

    def __prompt_internal(
            self,
            human_prompt: str,
            system_prompt: str | None = None,
            response_count: int = 1,
            temperature: float = 0.3,
            model_name: str | None = None,
            max_tokens: int | None = None,
    ) -> List[str]:
        """
        This function is used to get the response from the Openai LLM API.
        Args:
            human_prompt: Human prompt. Has role user.
            system_prompt: System prompt. Has role system. Treated as general instructions. Higher priority than human
            prompt.
            response_count: Count of responses. (Ho many responses for this one human and system prompt.)
            temperature: Temperature. Creativity and randomness of answer. (0-2)
            model_name: Name of OpenAI model used.
            max_tokens: Max count of tokens.

        Returns:
            List of answers based on human and system prompt.

        """
        result = []
        try:
            if system_prompt is None:
                messages = [{"role": "user", "content": human_prompt}]
            else:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": human_prompt}]
            client = OpenAI(api_key=self.model.api_key)
            if model_name is not None and 'o3' in model_name:
                response = client.chat.completions.create(
                    model=self.model.name if model_name is None else model_name,
                    messages=messages,
                    n=response_count,
                    max_completion_tokens=max_tokens if max_tokens is None else max_tokens,
                    timeout=300,
                    reasoning_effort="low"  # Can be "low", "medium", or "high"
                )
            else:
                response = client.chat.completions.create(
                    model=self.model.name if model_name is None else model_name,
                    messages=messages,
                    n=response_count,
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens is None else max_tokens,
                    timeout=120
                )
            if response.choices[0].finish_reason != 'stop':
                logger.error("!!! GPT was stopped because of: ")
                logger.error(response.choices[0].finish_reason)
                logger.warning("Adding incomplete answer to result")
                for choice in response.choices:
                    answer = choice.message.content
                    result.append(answer)
                if response.choices[0].finish_reason == 'length':
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    raise Exception(f"maximum context length exceeded: {prompt_tokens + completion_tokens} "
                                    f"{total_tokens}")

            else:
                for choice in response.choices:
                    answer = choice.message.content
                    result.append(answer)
        except Exception as exc:
            logger.error(exc)
        return result

    def get_llm_api_response_with_backup_special(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model_name: str | None = None,
            response_count: int = 1,
            temperature: float | None = None,
            get_multiple_answers: bool = True,
            output_format: OutputFormat = OutputFormat.NO_FORMAT,
            answer_type: AnswerType | None = None
    ) -> List[str] | List[StructuredOutput]:
        """
        General function to get the response from the Openai LLM API.
        Args:
            prompt: Human prompt. Has role user.
            system_prompt: System prompt. Has role system. Treated as general instructions. Higher priority than human
            prompt.
            response_count: Count of responses. (Ho many responses for this one human and system prompt.)
            temperature: Temperature. Creativity and randomness of answer. (0-2)
            model_name: Name of OpenAI model used.
            get_multiple_answers: Whether to return more than one answer. Should be True if response_count > 1.
            output_format: Output format from OutputFormat Enum.
            answer_type: Answer type from AnswerType Enum.

        Returns:
            List of answers based on human and system prompt.
        """
        result = ''
        try:
            if answer_type is None and output_format != OutputFormat.NO_FORMAT:
                raise Exception("Answer type is None")
            if not get_multiple_answers and response_count > 1:
                raise Exception(f"Response count is 1 for multiple answer method.")
            # if response_count == 1:
            #     raise Exception(f"Response count is 1 for multiple answer method.")
            # else:
            if output_format == OutputFormat.NO_FORMAT:
                responses = self.__prompt_internal(
                    system_prompt=system_prompt,
                    human_prompt=prompt,
                    response_count=response_count,
                    temperature=1.0 if temperature is None else temperature,
                    model_name=model_name
                )
            else:
                responses = self.get_structured_output(
                    human_prompt=prompt,
                    answer_type=answer_type,
                    system_prompt=system_prompt,
                    model_name=model_name,
                    response_count=response_count,
                    temperature=1.0 if temperature is None else temperature,
                    output_format=output_format
                )
            result = responses[:response_count]
        except Exception as e:
            logger.error(e)

        return result

    def get_structured_output(
            self,
            human_prompt: str,
            answer_type: AnswerType,
            system_prompt: str | None = None,
            model_name: str | None = None,
            response_count: int = 1,
            temperature: float = 0.0,
            output_format: OutputFormat = OutputFormat.STRUCTURED_COT
    ):
        """
        Gets Structured output from OpenAi LLM API.
        Args:
            human_prompt: Human prompt. Has role user.
            answer_type: Answer type from AnswerType Enum.
            system_prompt: System prompt. Has role system. Treated as general instructions. Higher priority than human
            prompt.
            model_name: Name of OpenAI model used.
            response_count: Count of responses. (Ho many responses for this one human and system prompt.)
            temperature: Temperature. Creativity and randomness of answer. (0-2)
            output_format: Output format from OutputFormat Enum.

        Returns:
            List of StructuredOutput answers in desired OutputFormat.
        """
        results: List[StructuredOutput] = []

        try:
            client = OpenAI(api_key=self.model.api_key)
            if answer_type == AnswerType.MULTIPLE_CHOICE:
                if output_format == OutputFormat.STRUCTURED_COT:
                    response_format = StructuredOutputModelMultipleChoice
                elif output_format == OutputFormat.STRUCTURED_ANSWER:
                    response_format = StructuredOutputModelMultipleChoiceOnlyChoice
                elif output_format == OutputFormat.STRUCTURED_EXTRA:
                    response_format = StructuredOutputModelMultipleChoiceExtra
                else:
                    raise Exception(f"Output format {output_format} is not yet implemented.")
            elif answer_type == AnswerType.NUMBER:
                if output_format == OutputFormat.STRUCTURED_COT:
                    response_format = StructuredOutputModelNumber
                elif output_format == OutputFormat.STRUCTURED_ANSWER:
                    response_format = StructuredOutputModelNumberOnlyNumber
                else:
                    raise Exception(f"Output format {output_format} is not yet implemented.")
            else:
                raise Exception(f"Answer type {answer_type.value} is not yet implemented.")
            if system_prompt is None:
                messages = [{"role": "user", "content": human_prompt}]
            else:
                messages = [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": human_prompt}]
            response = client.beta.chat.completions.parse(
                model=self.model.name if model_name is None else model_name,
                messages=messages,
                n=response_count,
                temperature=temperature,
                response_format=response_format,
                timeout=120
            )
            if response.choices[0].finish_reason != 'stop':
                logger.error("!!! GPT was stopped because of: ")
                logger.error(response.choices[0].finish_reason) # TODO save unfinished answer?
                if response.choices[0].finish_reason == 'length':
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    raise Exception(f"maximum context length exceeded: {prompt_tokens + completion_tokens} "
                                    f"{total_tokens}")
            else:
                for choice in response.choices:
                    answer_raw = choice.message.parsed
                    answer_obj = StructuredOutput(
                        solution_explanation=answer_raw.solution_explanation if
                        output_format != OutputFormat.STRUCTURED_ANSWER else "",
                        answer_as_letter=answer_raw.answer_as_letter if
                        answer_type == AnswerType.MULTIPLE_CHOICE else "",
                        answer_as_number=answer_raw.answer_as_number if
                        answer_type == AnswerType.NUMBER else None,
                        extracted_variables=answer_raw.extracted_variables if
                        output_format == OutputFormat.STRUCTURED_EXTRA else [],
                        steps_for_answer=answer_raw.steps_for_answer if
                        output_format == OutputFormat.STRUCTURED_EXTRA else [],
                    )
                    results.append(answer_obj)

        except Exception as exc:
            logger.error(exc) # exc.completion.choices[0].finish_reason == 'length'
        return results


    # TODO max tokens maybe makes answer not be infinite?

    # TODO error handling and retries if reason is length

