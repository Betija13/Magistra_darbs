from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
from typing import List


@dataclass_json
@dataclass
class RankingResults:
    chosen_answer: str | None = None
    answer_score: float | None = None
