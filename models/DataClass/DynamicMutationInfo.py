from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DynamicMutationInfo:
    id: int
    start_task_prompt: str
    mutated_task_prompt: str
    llm_answer: str
    correct_answer: bool
    reward_model_name: str | None
    reward_model_score_task_question: float | None
    reranking_score_task_question: float | None
    reward_model_score_answer_question: float | None
    reranking_score_answer_question: float | None
    mutation_prompt: str
    thinking_style: str
    question: str
    llm_temperature: float
    result_file: str = ""

