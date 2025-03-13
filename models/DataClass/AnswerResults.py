from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AnswerResults:
    llm_answer_unedited: str = ""
    correct: bool = False
    chosen_answer: str | None = None
    task_prompts_all: str | None = None
    task_prompts_majority: str | None = None


