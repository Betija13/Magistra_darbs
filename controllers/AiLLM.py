import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from openai.types.chat import ChatCompletion

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import openai
from openai import OpenAI
import requests
import tiktoken
from dataclasses_json import dataclass_json
from loguru import logger
from openai.lib.azure import AzureOpenAI
# from models.dataClass.PhraseVariations import PhraseVariations, PhraseVariationsModel
from openai import OpenAI
from iso639 import Lang
# from controllers.ControllerAiEmbeddingSentence import ControllerAiEmbeddingSentence
# from utils.config_and_key_utils import ConfigKeyUtils
# from utils.decorators.measure_function_speed import measure_function_speed_with_name
# from utils.discord_utils import DiscordLogger
import os
from typing import Dict
from models.DataClass.LLMModel import LLMModel
from models.DataClass.StructuredOutput import StructuredOutput, StructuredOutputModelMultipleChoice, \
    StructuredOutputModelMultipleChoiceOnlyChoice, StructuredOutputModelNumber, StructuredOutputModelNumberOnlyNumber
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

LLM_MAIN_MODEL = config.get('LLM_MAIN_MODEL')
LLM_MAIN_API_KEY = config.get('LLM_MAIN_API_KEY')


class ControllerAiLLM:
    def __init__(self):
        self.model: LLMModel = LLMModel(
            name='gpt-4o', #LLM_MAIN_MODEL,
            api_key=LLM_MAIN_API_KEY
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
            get_multiple_answers: bool = False
    ) -> str | List[str]:
        result = ''
        try:
            if response_count == 1:
                result = self.__prompt_internal(
                    system_prompt=system_prompt,
                    human_prompt=prompt,
                    response_count=response_count,
                    temperature=0.3 if temperature is None else temperature,
                    model_name=model_name
                )[0]
            else:
                responses = self.__prompt_internal(
                    system_prompt=system_prompt,
                    human_prompt=prompt,
                    response_count=response_count,
                    temperature=1.0 if temperature is None else temperature,
                    model_name=model_name
                )
                if not get_multiple_answers:
                    logger.error(f"Get multiple answers is False but response count is not 1.")
                else:
                    result = responses[:response_count]
        except Exception as e:
            logger.error(e)

        return result

    def get_structured_output_multiple_choice(
            self,
            human_prompt: str,
            answer_type: str,
            system_prompt: str | None = None,
            model_name: str | None = None,
            response_count: int = 1,
            temperature: float = 0.0,
            only_answer: bool = False
    ):
        result: List[StructuredOutput] = []

        try:
            client = OpenAI(api_key=self.model.api_key)
            if answer_type == AnswerType.MULTIPLE_CHOICE.value:
                response_format = StructuredOutputModelMultipleChoiceOnlyChoice if only_answer else StructuredOutputModelMultipleChoice
            elif answer_type == AnswerType.NUMBER.value:
                response_format = StructuredOutputModelNumberOnlyNumber if only_answer else StructuredOutputModelNumber
            else:
                raise Exception(f"Answer type {answer_type} is not yet implemented.")
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
                        solution_explanation=answer_raw.solution_explanation if not only_answer else "",
                        answer_as_letter=answer_raw.answer_as_letter if answer_type == AnswerType.MULTIPLE_CHOICE.value else "",
                        answer_as_number=answer_raw.answer_as_number if answer_type == AnswerType.NUMBER.value else 0.0,
                    )
                    result.append(answer_obj)

        except Exception as exc:
            logger.error(exc)
        return result


    # TODO max tokens maybe makes answer not be infinite?

