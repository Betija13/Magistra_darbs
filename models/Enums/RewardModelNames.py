from enum import Enum
import os
from loguru import logger
from dotenv import dotenv_values

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


class RewardModelNames(Enum):
    INTERNLM_1_8_B = "internlm/internlm2-1_8b-reward"
    INTERNLM_7_B = "internlm/internlm2-7b-reward" # Did not work properly on my device
    DEBERTA_V3_2 = "OpenAssistant/reward-model-deberta-v3-large-v2"
    LLM_GEMINI = "gemini-2.0-flash"
    RERANK_MODEL = FACT_RANKING_MODEL
    BLENDER_PRM = 'llm-blender/PairRM'
    SAFAIRXC = 'sfairXC/FsfairX-LLaMA3-RM-v0.1'
    QRM_8B_V2 = 'nicolinho/QRM-Llama3.1-8B-v2'
    QRM_8B = "nicolinho/QRM-Llama3.1-8B"
    QRM_8B_P = "nicolinho/QRM-Llama3-8B"
    GRM = 'Ray2333/GRM-Llama3-8B-rewardmodel-ft'
    SKYWORK = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    ARMORM = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    URM = "LxzGordon/URM-LLaMa-3.1-8B"
    PAIRRM_B = "maywell/Better-PairRM"
    EURUS = "openbmb/Eurus-RM-7b"
