from collections import Counter
from typing import List, Dict
from loguru import logger
import re
import os
import requests
import json
import torch
import llm_blender
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
import psutil
from dotenv import dotenv_values
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, pipeline
from controllers.AiLLM import ControllerAiLLM
from models.DataClass.SplitOptions import SplitOptions
from models.DataClass.RankingResults import RankingResults
from models.Enums.RewardModelNames import RewardModelNames
from models.DataClass.StructuredOutputRanking import StructuredOutputRanking, OutputRankingScores, OutputRankingBestIdx

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
GPU_MEM = config.get('GPU_MEM')
CPU_MEM = config.get('CPU_MEM')
if FACT_RANKING_MODEL is None or RERANK_URL is None:
    logger.critical(f"Missing variable in .env file. {FACT_RANKING_MODEL=}\t{RERANK_URL=}")


class RewardMethods:

    def __init__(self, ai_llm: ControllerAiLLM | None = None, reward_name: RewardModelNames | None = None) -> None:
        self.ai_llm = ai_llm if ai_llm else ControllerAiLLM()
        self.reward_name = reward_name.value if reward_name else None
        self.reward_model = None
        self.tokenizer = None
        self.kwargs = None
        self.device = 'cpu'

        if self.reward_name:
            self.init_reward_model(reward_model_name=reward_name)


    def init_reward_model(self, reward_model_name: RewardModelNames) -> None: # TODO sakārtot ar kwargs
        """
        Initialize the reward model and tokenizer with the specified model name.

        Args:
            reward_model_name: The name of the reward model to load.

        """
        torch.cuda.empty_cache()
        max_memory = {0: GPU_MEM, "cpu": CPU_MEM} if (GPU_MEM and CPU_MEM) else None
        free_mem, total_mem = torch.cuda.mem_get_info(device=0)
        free_mem_gb = free_mem / 1024**3
        total_mem_gb = total_mem / 1024**3
        mem = psutil.virtual_memory()
        device_map = None
        if torch.cuda.is_available():
            device_map = "auto"
            self.device = 'cuda:0'
        else:
            max_memory = None
        self.reward_name = reward_model_name.value
        if reward_model_name == RewardModelNames.DEBERTA_V3_2: # TODO adapt for GPU?
            # https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2
            # https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large reward_name = "OpenAssistant/reward-model-deberta-v3-large"
            self.reward_model, self.tokenizer = AutoModelForSequenceClassification.from_pretrained(
                self.reward_name), AutoTokenizer.from_pretrained(self.reward_name)
        elif reward_model_name in [RewardModelNames.INTERNLM_1_8_B, RewardModelNames.INTERNLM_7_B]:
            # https://xtuner.readthedocs.io/en/latest/reward_model/overview.html
            # https://huggingface.co/internlm/internlm2-1_8b-reward
            # https://huggingface.co/internlm/internlm2-20b-reward
            self.reward_model = AutoModel.from_pretrained(
                self.reward_name,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name, trust_remote_code=True)


        elif reward_model_name == RewardModelNames.BLENDER_PRM:
            # https://huggingface.co/llm-blender/PairRM
            self.reward_model = llm_blender.Blender()
            self.reward_model.loadranker(self.reward_name)  # load PairRM
        elif reward_model_name == RewardModelNames.PAIRRM_B:
            # https://huggingface.co/mightbe/Better-PairRM
            self.reward_model = DebertaV2PairRM.from_pretrained(self.reward_name, device_map=self.device, max_memory=max_memory).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)

        elif reward_model_name == RewardModelNames.SAFAIRXC:
            #https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_name,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16
            )

            # Now create the pipeline with the pre-loaded model and tokenizer
            self.reward_model = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=self.tokenizer,
            )

            self.kwargs = {
                "return_all_scores": True,
                "function_to_apply": "none",
                "batch_size": 1
            }
        elif reward_model_name in [RewardModelNames.QRM_8B_V2, RewardModelNames.QRM_8B, RewardModelNames.QRM_8B_P,
                                   RewardModelNames.ARMORM, RewardModelNames.URM]:
            # https://huggingface.co/nicolinho/QRM-Llama3.1-8B-v2
            # https://huggingface.co/nicolinho/QRM-Llama3.1-8B
            # https://huggingface.co/nicolinho/QRM-Llama3-8B
            # https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
            kwargs = dict(
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True
            )
            if reward_model_name in [RewardModelNames.QRM_8B_V2, RewardModelNames.ARMORM]:
                kwargs['torch_dtype'] = torch.bfloat16

            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_name, **kwargs
            )
            if reward_model_name in [RewardModelNames.URM, RewardModelNames.QRM_8B_P]:
                self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name, use_fast=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
        elif reward_model_name == RewardModelNames.GRM:
            # https://huggingface.co/Ray2333/GRM-Llama3-8B-rewardmodel-ft
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_name, torch_dtype=torch.float16,
                device_map=device_map, max_memory=max_memory
            )
        elif reward_model_name == RewardModelNames.SKYWORK:
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.reward_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                max_memory=max_memory,
                # attn_implementation="flash_attention_2",
                num_labels=1,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
        elif reward_model_name == RewardModelNames.EURUS:
            # https://huggingface.co/openbmb/Eurus-RM-7b
            self.tokenizer = AutoTokenizer.from_pretrained(self.reward_name)
            self.reward_model = AutoModel.from_pretrained(
                self.reward_name, trust_remote_code=True, max_memory=max_memory, device_map=device_map
            )

        # https://huggingface.co/infly/INF-ORM-Llama3.1-70B
        # Best from https://huggingface.co/spaces/allenai/reward-bench but model size aprox 145 GB




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
    ) -> str | None: # TODO change for Gemini
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
        """
        Use the reranking model to evaluate the answers and return the best one.
        Args:
            question: Question to which the answer is being evaluated.
            answer_options: Answer options to be evaluated.

        Returns:
            List of answer options with their scores. Best are at the start.
        """
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
            response.raise_for_status()
            rankings = json.loads(response.text).get('rankings')
            # TODO adapt to a Dataclass
        except Exception as e:
            logger.error(f"Failed to call rerank model API: {e}")
            # logger.exception(e)
        return rankings

    def get_reranking_model_best_answer(self, question: str, answer_options: List[str]) -> RankingResults:
        """
        Use the reranking model to evaluate the answers and return the best one.
        Args:
            question: Question to which the answer is being evaluated.
            answer_options: Answer options to be evaluated.

        Returns:
            The best answer according to the evaluation of the reranking model and the score for the best answer.
        """
        result_answer = RankingResults()
        try:
            tries_count = 0
            successful = False
            max_size = 1000
            while not successful and tries_count < 5:
                split_options = self.split_options_for_rerank(answer_options=answer_options, max_size=max_size)
                split_options_answers = split_options.answers_split
                answer_option_idxes = split_options.answer_idxes
                try:
                    results_rerank = self.reranking_model(question, split_options_answers)
                except Exception as e:
                    logger.error(e)
                    results_rerank = []
                tries_count += 1
                if len(results_rerank) == len(split_options_answers):
                    successful = True
                else:
                    max_size -= 150
            if not successful:
                raise Exception("Reranking model failed to return all results")
            best_idx_split = results_rerank[0]['index']
            result_answer.answer_score = results_rerank[0]['logit']
            best_idx_original = answer_option_idxes[best_idx_split]
            result_answer.chosen_answer = answer_options[best_idx_original]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reranking_model_score(self, question: str, answer_option: str) -> float | None:
        """
        Get the score of the answer option using the reranking model.
        Args:
            question: Question that is being answered.
            answer_option: Answer to the Question.

        Returns:
            Answer options score for the question.
        """
        result_answer = None
        total_tries = 0
        max_size = 1000
        while result_answer is None and total_tries < 5:
            try:
                answer_options = self.split_options_for_rerank(answer_options=[answer_option], max_size=max_size)
                results_rerank = self.reranking_model(question, answer_options.answers_split)
                result_answer = results_rerank[0]['logit']
            except Exception as e:
                logger.error(e)
                max_size -= 100
        return result_answer

    def split_options_for_rerank(self, answer_options: List[str], max_size: int = 1000) -> SplitOptions:
        """
        Split the answer options into smaller parts for reranking. Also saves the original answer idxes
        Args:
            answer_options: list of answer options.
            max_size: max size of single answer option or by what amount of characters is the option being split.
                By default, 1000.

        Returns:
            SplitOptions object that contains split answers and their idxes of original list.

        """
        split_answers = SplitOptions()
        try:
            answer_options_final = []
            answer_option_idxes = []
            for idx, answer_option in enumerate(answer_options):
                answer_options_split = [answer_option]
                if len(answer_option) > max_size:
                    parts = len(answer_option) // max_size + 1
                    len_part = len(answer_option) / parts
                    answer_options_split = [answer_option[int(i * len_part):int((i + 1) * len_part)] for i in
                                            range(parts)]
                for _ in range(len(answer_options_split)):
                    answer_option_idxes.append(idx)
                answer_options_final.extend(answer_options_split)
            split_answers.answers_split = answer_options_final
            split_answers.answer_idxes = answer_option_idxes

        except Exception as e:
            logger.error(e)
        return split_answers

    def get_reward_model_deberta_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                inputs = self.tokenizer(question, answer, return_tensors='pt')
                score = self.reward_model(**inputs).logits[0].cpu().detach()
                scores.append(float(score))
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_deberta_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            inputs = self.tokenizer(question, answer_option, return_tensors='pt')
            score = self.reward_model(**inputs).logits[0].cpu().detach()
            score = float(score[0])
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_internlm_scores(
            self,
            answer_options: List[str],
            question: str
    ) -> List[float] | None:
        """
            Gets reward scores for each answer option
            Args:
                answer_options: List of answer options.
                question: Question to be asked.
            Returns:
                List of scores for each answer option.
        """
        scores = None
        try:
            chats = []
            for answer_option in answer_options:
                chat_n = [
                    {"role": "user", "content": question},
                    {"role": "assistant",
                     "content": answer_option}
                ]
                chats.append(chat_n)
            scores = self.reward_model.get_scores(self.tokenizer, chats)
        except Exception as e:
            logger.error(e)
        return scores

    def get_reward_model_blender_prm_scores(
            self,
            answer_options: List[str],
            question: str
    ) -> List[float] | None:
        """
            Gets reward scores for each answer option
            Args:
                answer_options: List of answer options.
                question: Question to be asked.
            Returns:
                List of scores for each answer option.
        """
        scores = None
        try:
            inputs = [question]
            candidates_texts = [answer_options]
            ranks = self.reward_model.rank(inputs, candidates_texts, return_scores=True, batch_size=1)
            scores = list(ranks[0])
        except Exception as e:
            logger.error(e)
        return scores

    def get_reward_model_internlm_best_answer(self, answer_options: List[str], question: str) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.
        Returns:
            The best answer according to the evaluation of the reward model and its score.
        """
        result_answer = RankingResults()
        try:
            scores = self.get_reward_model_internlm_scores(answer_options=answer_options, question=question)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_blender_prm_best_answer(self, answer_options: List[str], question: str) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.
        Returns:
            The best answer according to the evaluation of the reward model and its score.
        """
        result_answer = RankingResults()
        try:
            scores = self.get_reward_model_blender_prm_scores(answer_options=answer_options, question=question)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_safairxc_scores(self, answer_options: List[str], question: str) -> List[float] | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_options: List of answer options.
            Returns:
                Score of the answer option.
        """
        scores = None
        try:
            test_texts = []
            for answer_n in answer_options:
                chat = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_n}
                ]

                test_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(
                        self.tokenizer.bos_token,
                        "")
                test_texts.append(test_text)
            pipe_outputs = self.reward_model(test_texts, **self.kwargs)
            scores = [output[0]["score"] for output in pipe_outputs]
        except Exception as e:
            logger.error(e)
        return scores

    def get_reward_model_safairxc_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = self.get_reward_model_safairxc_scores(answer_options=answer_options, question=question)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_qrm_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            messages = [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer_option}]
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.reward_model(input_ids)
                reward = output.score.cpu().float()
                score = float(reward)
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_qrm_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_qrm_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_grm_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            message = [
                {'role': 'user',
                 'content': question},
                {'role': 'assistant',
                 'content': answer_option}
            ]
            message_template = self.tokenizer.apply_chat_template(message, tokenize=False)

            kwargs = {"padding": 'longest', "truncation": True, "return_tensors": "pt"}
            tokens = self.tokenizer.encode_plus(message_template, **kwargs)

            with torch.no_grad():
                reward_tensor = self.reward_model(tokens["input_ids"][0].view(1, -1).to(self.device),
                                             attention_mask=tokens["attention_mask"][0].view(1, -1).to(self.device))[0]
                score = reward_tensor.cpu().detach().item()
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_grm_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_grm_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_skywork_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            conv1 = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_option}]
            conv1_tokenized = self.tokenizer.apply_chat_template(conv1, tokenize=True, return_tensors="pt").to(self.device)

            # Get the reward scores
            with torch.no_grad():
                score = self.reward_model(conv1_tokenized).logits[0][0].item()
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_skywork_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_skywork_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_armorm_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_option}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.device)
            with torch.no_grad():
                output = self.reward_model(input_ids)
                score = output.score.float().item()
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_armorm_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_armorm_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_urm_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer_option}]
            resp1 = self.tokenizer.apply_chat_template(messages, tokenize=False)
            resp1 = self.tokenizer(resp1, return_tensors="pt").to(self.device)
            with torch.no_grad():
                score = self.reward_model(resp1['input_ids'], attention_mask=resp1['attention_mask']).logits[0][0].item()

        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_urm_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_urm_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def pairrm_tokenize_pair(
            self,
            sources:List[str],
            candidate1s:List[str],
            candidate2s:List[str],
            source_max_length=2030,
            candidate_max_length=670
    ):
        """
        Function from https://huggingface.co/mightbe/Better-PairRM
        """
        source_prefix = "<|source|>"
        cand1_prefix = "<|candidate1|>"
        cand2_prefix = "<|candidate2|>"
        ids = []
        assert len(sources) == len(candidate1s) == len(candidate2s)
        max_length = source_max_length + 2 * candidate_max_length
        for i in range(len(sources)):
            source_ids = self.tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
            candidate_max_length = (max_length - len(source_ids)) // 2
            candidate1_ids = self.tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True)
            candidate2_ids = self.tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True)
            ids.append(source_ids + candidate1_ids + candidate2_ids)
        encodings = self.tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
        return encodings

    def get_reward_model_pairrm_b_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            answer_chosen = answer_options.pop()
            for answer_comparing in answer_options:
                encodings = self.pairrm_tokenize_pair([question], [answer_comparing], [answer_chosen])
                encodings = {k: v.to(self.reward_model.device) for k, v in encodings.items()}
                outputs = self.reward_model(**encodings)
                # logits = outputs.logits.tolist()
                comparison_results = outputs.logits > 0
                comparison_bool = bool(comparison_results)
                if comparison_bool:
                    answer_chosen = answer_comparing
                del outputs
            result_answer.chosen_answer = answer_chosen
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_eurus_score(self, answer_option: str, question: str) -> float | None:
        """
            Get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            text_input = f"[INST] {question} [/INST] {answer_option}"
            with torch.no_grad():
                inputs = self.tokenizer(text_input, return_tensors="pt")
                score = self.reward_model(**inputs).item()
        except Exception as e:
            logger.error(e)
        return score

    def get_reward_model_eurus_best_answer(
            self,
            answer_options: List[str],
            question: str
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            scores = []
            for answer in answer_options:
                score = self.get_reward_model_eurus_score(answer_option=answer, question=question)
                scores.append(score)
            max_score = max(scores)
            result_answer.answer_score = max_score
            result_answer.chosen_answer = answer_options[scores.index(max_score)]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_best_answer(
            self,
            answer_options: List[str],
            question: str,
            reward_name: RewardModelNames | None = None
    ) -> RankingResults:
        """
        Use the reward model to evaluate the answers and return the best one.
        Args:
            answer_options: List of answer options.
            question: Question to be asked.
            reward_name: Name of the reward model.

        Returns:
            The best answer according to the evaluation of the reward model and the score for that answer.

        """
        result_answer = RankingResults()
        try:
            if reward_name and reward_name != self.reward_name:
                self.init_reward_model(reward_name)
            if self.reward_name is None or self.reward_model is None:
                raise Exception(f"Reward model not initialized")
            if self.reward_name in [RewardModelNames.DEBERTA_V3_2.value]:
                result_answer = self.get_reward_model_deberta_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.INTERNLM_1_8_B.value, RewardModelNames.INTERNLM_7_B.value]:
                result_answer = self.get_reward_model_internlm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.BLENDER_PRM.value]:
                result_answer = self.get_reward_model_blender_prm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.SAFAIRXC.value]:
                result_answer = self.get_reward_model_safairxc_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.QRM_8B_V2.value, RewardModelNames.QRM_8B.value, RewardModelNames.QRM_8B_P.value]:
                result_answer = self.get_reward_model_qrm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.GRM.value]:
                result_answer = self.get_reward_model_grm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.SKYWORK.value]:
                result_answer = self.get_reward_model_skywork_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.ARMORM.value]:
                result_answer = self.get_reward_model_armorm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.URM.value]:
                result_answer = self.get_reward_model_urm_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.PAIRRM_B.value]:
                result_answer = self.get_reward_model_pairrm_b_best_answer(
                    answer_options=answer_options, question=question
                )
            elif self.reward_name in [RewardModelNames.EURUS.value]:
                result_answer = self.get_reward_model_eurus_best_answer(
                    answer_options=answer_options, question=question
                )

            else:
                raise Exception(f"Unsupported self.reward_name: {self.reward_name}")

        except Exception as e:
            logger.error(e)
        return result_answer

    def get_reward_model_score(self, answer_option: str, question: str, reward_name: RewardModelNames | None = None) -> float | None:
        """
            General function to get the score of the answer option using the reward model.
            Args:
                question: Question to be asked.
                answer_option: Answer option to be evaluated.
                reward_name: Name of the reward model. (Optional, to change model)
            Returns:
                Score of the answer option.
        """
        score = None
        try:
            if reward_name and reward_name != self.reward_name:
                self.init_reward_model(reward_name)
            if self.reward_name is None or self.reward_model is None:
                raise Exception(f"Reward model not initialized")
            if self.reward_name in [RewardModelNames.DEBERTA_V3_2.value]:
                score = self.get_reward_model_deberta_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.INTERNLM_1_8_B.value, RewardModelNames.INTERNLM_7_B.value]:
                scores = self.get_reward_model_internlm_scores(answer_options=[answer_option], question=question)
                score = scores[0]
            elif self.reward_name in [RewardModelNames.BLENDER_PRM.value]:
                scores = self.get_reward_model_blender_prm_scores(answer_options=[answer_option], question=question)
                score = scores[0]
            elif self.reward_name in [RewardModelNames.SAFAIRXC.value]:
                scores = self.get_reward_model_safairxc_scores(answer_options=[answer_option], question=question)
                score = scores[0]
            elif self.reward_name in [RewardModelNames.QRM_8B_V2.value, RewardModelNames.QRM_8B.value, RewardModelNames.QRM_8B_P.value]:
                score = self.get_reward_model_qrm_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.GRM.value]:
                score = self.get_reward_model_grm_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.SKYWORK.value]:
                score = self.get_reward_model_skywork_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.ARMORM.value]:
                score = self.get_reward_model_armorm_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.URM.value]:
                score = self.get_reward_model_urm_score(answer_option=answer_option, question=question)
            elif self.reward_name in [RewardModelNames.EURUS.value]:
                score = self.get_reward_model_eurus_score(answer_option=answer_option, question=question)

            else:
                raise Exception(f"Unsupported self.reward_name: {self.reward_name}")
        except Exception as e:
            logger.error(e)
        return score

    def get_llm_best_answer_reranker(self, answer_options: List[str], question: str) -> RankingResults:
        """
        Use the LLM to evaluate the answers and return the best one. Using LLM as reranker, giving multiple
        answer options and making it give scores for all of them.
        Args:
            answer_options: Answer options being evaluated.
            question: Question for which the answers are evaluated.

        Returns:
            The best answer according to the evaluation of the LLM and the score for that answer.
        """
        result_answer = RankingResults()
        try:
            system_prompt = "Provide scores for each of answer options to the question based on your analysis. " \
                            "Use logical reasoning and contextual understanding to determine the most appropriate " \
                            "answer for question and give that the highest score (10). "
            answers_str = '\n\n'.join([f"ANSWER {idx}:\n{answer}" for idx, answer in enumerate(answer_options)])
            human_prompt = f'Question:\n```\n{question}\n```\nAnswer options:\n```\n{answers_str}\n```'
            llm_answer_ranking = self.ai_llm.prompt_gemini_ranking(
                system_prompt=system_prompt, human_prompt=human_prompt, output_format=OutputRankingScores
            )
            rankings = llm_answer_ranking.answer_rankings
            rankings_sorted = sorted(rankings, key=lambda r: r.answer_score, reverse=True)
            best_answer = rankings_sorted[0]
            chosen_idx = best_answer.answer_idx
            score = best_answer.answer_score
            result_answer.answer_score = score
            result_answer.chosen_answer = answer_options[chosen_idx]
        except Exception as e:
            logger.error(e)
        return result_answer

    def get_llm_best_answer_best_idx(self, answer_options: List[str], question: str) -> RankingResults:
        """
        Use the LLM to evaluate the answers and return the best one. Getting only the best answer's idx as answer.
        Args:
            answer_options: Answer options being evaluated.
            question: Question for which the answers are evaluated.

        Returns:
            The best answer according to the evaluation of the LLM and the score for that answer.
        """
        result_answer = RankingResults()
        try:
            system_prompt = "Provide the index of the best answer based on your analysis. Use logical reasoning " \
                            "and contextual understanding to determine the most appropriate answer. "
            answers_str = '\n\n'.join([f"ANSWER {idx}:\n{answer}" for idx, answer in enumerate(answer_options)])
            human_prompt = f'Question:\n```\n{question}\n```\nAnswer options:\n```\n{answers_str}\n```'
            llm_answer_ranking = self.ai_llm.prompt_gemini_ranking(
                system_prompt=system_prompt, human_prompt=human_prompt, output_format=OutputRankingBestIdx
            )
            chosen_idx = llm_answer_ranking.best_idx
            score = llm_answer_ranking.best_answer_score
            result_answer.answer_score = score
            result_answer.chosen_answer = answer_options[chosen_idx]
        except Exception as e:
            logger.error(e)
        return result_answer
