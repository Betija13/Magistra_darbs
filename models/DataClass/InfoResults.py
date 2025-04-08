from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class InfoResults:
    id: int = 0
    date: str = ""
    dataset_name: str = ""
    result_file_name: str = ""
    method: str = ""
    finished: bool = False
    count: int = 0
    accuracy: float = 0.0
    system_prompt: str = ""
    human_prompt: str = ""
    temperature: float = 0.0
    response_count: int = 1
    reward_method: str | None = None
    llm_model: str = ""
    percentage_of_short_answers: float | None = None
