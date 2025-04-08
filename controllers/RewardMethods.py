from collections import Counter
from typing import List, Dict
from loguru import logger
import re
import os
import requests
import json
from dotenv import dotenv_values
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from controllers.AiLLM import ControllerAiLLM

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

FACT_RANKING_MODEL = config.get('FACT_RANKING_MODEL')
RERANK_URL = config.get('RERANK_URL')

class RewardMethods:

    def __init__(self, ai_llm: ControllerAiLLM | None = None):
        self.ai_llm = ai_llm if ai_llm else ControllerAiLLM()
        self.reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2" # https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2
        # https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large reward_name = "OpenAssistant/reward-model-deberta-v3-large"
        self.rank_model, self.tokenizer = AutoModelForSequenceClassification.from_pretrained(
            self.reward_name), AutoTokenizer.from_pretrained(self.reward_name)

    def init_reward_model(self, reward_model_name: str):
        self.reward_name = reward_model_name
        self.rank_model, self.tokenizer = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name), AutoTokenizer.from_pretrained(reward_model_name)
    @staticmethod
    def majority_element(answer_options: List[str]) -> str | None:
        """
        Find the majority element (the most repeated answer) in the list of answers.
        Args:
            answer_options: List of answer options.

        Returns:
            Majority element if it exists, otherwise None. None is also returned if all the answers are unique.
        """
        if not answer_options:
            return None

        count = Counter(answer_options)
        majority_count = len(answer_options) // 2

        for num, cnt in count.items():
            if cnt > majority_count:
                return num

        return None

    def another_llm(
            self,
            answer_options: List[str],
            scorer_model_name: str | None = None,
            choosing_prompt: bool = False,
    ) -> str | None:
        """
        Use another LLM to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            scorer_model_name: name of the llm used for scoring.
            choosing_prompt: if the scoring is choosing prompt or answer. True for prompt, False for answer.

        Returns:
            The best answer according to the evaluation of another LLM.
        """
        result_answer = None
        try:
            all_answers = '\n-----***-----\n'.join(answer_options)
            scores = []
            for answer_option in answer_options:
                if choosing_prompt:
                    human_prompt = f"Given prompt:\n```\n{answer_option}\n```\nAll prompts:\n```\n{all_answers}\n```\n"
                    system_prompt = "On a scale from 1 to 10, rate the given prompt in comparison to other prompts."
                else:
                    human_prompt = f"Given answer:\n```\n{answer_option}\n```\nAll answers:\n```\n{all_answers}\n```\n"
                    system_prompt = "On a scale from 1 to 10, rate the given answer in comparison to other answers."
                if scorer_model_name is None:
                    scorer_model_name = "gpt-4o" if self.ai_llm.model.name != "gpt-4o" else 'o3-mini'
                response_count = 1
                temperature = 0
                get_multiple_answers = False
                answer_score = self.ai_llm.get_llm_api_response_with_backup_special(
                    prompt=human_prompt, system_prompt=system_prompt, model_name=scorer_model_name,
                    response_count=response_count, temperature=temperature, get_multiple_answers=get_multiple_answers
                )
                numbers_in_answer = re.findall(r'-?\d+\.\d+|-?\d+', answer_score)
                scores.append(float(numbers_in_answer[0]) if numbers_in_answer else 0)
            max_score = max(scores)
            result_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def reranking_model(self, question: str, answer_options: List[str]) -> List[Dict[str, int]]: # TODO dataclass
        rankings = []
        try:
            invoke_url = RERANK_URL
            headers = {
                "Accept": "application/json",
            }
            # Sets payload data, ignores empty strings to not trigger API 'string_too_short' validation error
            passages_payload = [{"text": text} for text in answer_options if len(text.strip()) > 0]
            payload = {
                "model": FACT_RANKING_MODEL,
                "query": {
                    "text": question
                },
                "passages": passages_payload
            }
            session = requests.Session()
            response = session.post(invoke_url, headers=headers, json=payload, timeout=(30, 60))
            logger.info(f"Ranking API response time: {response.elapsed.total_seconds()}")
            response.raise_for_status()
            rankings = json.loads(response.text).get('rankings')
        except Exception as e:
            logger.error("Failed to call rerank model API")
            logger.exception(e)
        return rankings

    def get_reranking_model_score(self, question: str, answer_options: List[str]) -> str | None:
        result_answer = None
        try:
            results_rerank = self.reranking_model(question, answer_options)
            result_answer = answer_options[results_rerank[0]['index']]
        except Exception as e:
            logger.error(e)
        return result_answer

    def reward_model(self, answer_options: List[str], question: str, reward_name: str | None = None) -> str | None:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.
            reward_name: Name of the reward model.

        Returns:
            The best answer according to the evaluation of the reward model.

        """
        result_answer = None
        try:
            if reward_name and reward_name != self.reward_name:
                self.init_reward_model(reward_name)
            scores = []
            for answer in answer_options:
                inputs = self.tokenizer(question, answer, return_tensors='pt')
                score = self.rank_model(**inputs).logits[0].cpu().detach()
                scores.append(float(score))
            max_score = max(scores)
            result_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

if __name__ == "__main__":
    reward_methods = RewardMethods()
    start_prompt = 'Solve the multiple choice math word problem, ensuring you provide a detailed explanation for the answer. Choose from the options (A), (B), (C), (D), or (E).'
    created_my_prompts = [
        "Elaborate on your reasoning process to determine the correct answer for the math word problem from options (A), (B), (C), (D), or (E).",
        "Break down and solve the math word problem step-by-step, clarifying your reasoning, and select the correct option from (A), (B), (C), (D), or (E).",
        "Explain the solution to the math problem thoroughly, clearly selecting the correct option from (A) to (E).",
        "Put your math cape on, rescue the answer from the jaws of indecision, and reveal whether it's A, B, C, D, or E!",
        "Pick a letter and pray that math agrees with you.",
        "Break down the math word problem step-by-step and select the correct option: (A), (B), (C), (D), or (E).",
        "Pick the correct multiple choice math answer while explaining why it's right, from choices (A) to (E).",
        "Select the correct answer for the math problem and explain your reasoning briefly."

    ]
    # result = reward_methods.another_llm(created_my_prompts, choosing_prompt=True)
    # logger.info("Result from another LLM:")
    # logger.info(result)
    # result2 = reward_methods.get_reranking_model_score(start_prompt, created_my_prompts)
    # logger.info("Result from reranking model:")
    # logger.info(result2)
    result3 = reward_methods.reward_model(created_my_prompts, start_prompt)
    logger.info("Result from reward model:")
    logger.info(result3)





