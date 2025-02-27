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

from dotenv import dotenv_values
from loguru import logger

GPT_VERSION_FAST = "gpt-3.5-turbo"
GPT_4_TURBO = 'gpt-4-turbo'
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
            name=LLM_MAIN_MODEL,
            api_key=LLM_MAIN_API_KEY
        )

    def __prompt_internal(
            self,
            system_prompt: str,
            human_prompt: str,
            response_count: int = 1,
            temperature: float = 0.3,
            model_name: str | None = None,
            max_tokens: int | None = None,
    ) -> List[str]:
        result = []
        try:
            client = OpenAI(api_key=self.model.api_key)
            response = client.chat.completions.create(
                model=self.model.name if model_name is None else model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": human_prompt}],
                n=response_count,
                temperature=temperature,
                max_tokens=max_tokens,
                # reasoning_effort="medium"  # Can be "low", "medium", or "high"
            )
            if response.choices[0].finish_reason != 'stop':
                logger.error("!!! GPT was stopped because of: ")
                logger.error(json.dumps(response, indent=4))
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
            logger.exception(exc)
        return result

    def get_llm_api_response_with_backup_special(
            self,
            system_prompt: str,
            prompt: str,
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

