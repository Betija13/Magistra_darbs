from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AnswerResults:
    llm_answer_unedited: str = ""
    correct: bool = False
    chosen_answer: str | None = None
    task_prompts_all: str | None = None
    task_prompts_majority: str | None = None
    task_prompts_correct: str | None = None
    task_system_prompts: List[str] = field(default_factory=list)


