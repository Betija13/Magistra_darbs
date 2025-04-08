from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class NumericResults:
    accuracy_score: float | None = None
    percentage_of_short_answers: float | None = None
