from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DataResults:
    id: int = 0
    quid: str | None = None
    question: str = ""
    true_answer: str = ""
    llm_answer: str = ""
    correct: bool = False
    llm_answer_chosen: str | None = None
    reward_method: str | None = None
    reward_score: str | None = None


