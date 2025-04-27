from enum import Enum


class RewardMethod(Enum):
    MAJOR = 'MAJOR_ANSWER'
    RERANK = 'RERANK'
    REWARD_M = 'REWARD_MODEL'
    LLM_O = 'ANOTHER_LLM'
