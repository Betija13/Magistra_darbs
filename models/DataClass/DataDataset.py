from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DataDataset:
    id: int = 0
    quid: str | None = None
    question: str = ""
    answer: str = ""
    answer_type: str = ""
    choices: str | None = None
    answer_word: str | None = None
    facts: str | None = None


