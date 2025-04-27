from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class AnswerExtraction:
    answer_llm_unedited: str | None = None
    processed_answer: str | None = None
    correct: bool = False
