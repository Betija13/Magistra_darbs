from pydantic.dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
from typing import List


@dataclass_json
@dataclass
class SplitOptions:
    answers_split: List[str] = field(default_factory=list)
    answer_idxes: List[int] = field(default_factory=list)
