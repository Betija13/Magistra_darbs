from enum import Enum


class RewardMethod(Enum):
    MAJOR = 'MAJOR_ANSWER'
    RERANK = 'RERANK'
    REWARD_M = 'REWARD_MODEL'
    LLM_O_R = 'ANOTHER_LLM_RERANKING'
    LLM_O_B_I = 'ANOTHER_LLM_BEST_IDX'
    CORRECT_A = 'CORRECT_ANSWER'
