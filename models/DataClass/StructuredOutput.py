from pydantic import BaseModel
from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List
from dataclasses_json import dataclass_json
from enum import Enum


class LetterChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    # E = "E" # TODO remove


class StructuredOutputModelMultipleChoice(BaseModel):
    solution_explanation: str  # think step by step, solution explanation, extract variables
    answer_as_letter: LetterChoice


class StructuredOutputModelMultipleChoiceExtra(BaseModel):
    extracted_variables: List[str] = field(default_factory=list)  # extracted variables
    steps_for_answer: List[str] = field(default_factory=list)  # steps for answer
    solution_explanation: str  # think step by step, solution explanation, extract variables
    answer_as_letter: LetterChoice


class StructuredOutputModelMultipleChoiceOnlyChoice(BaseModel):
    answer_as_letter: LetterChoice


class StructuredOutputModelNumber(BaseModel):
    solution_explanation: str  # think step by step, solution explanation, extract variables
    answer_as_number: float


class StructuredOutputModelNumberOnlyNumber(BaseModel):
    answer_as_number: float


@dataclass_json
@dataclass
class StructuredOutput:
    solution_explanation: str = ""  # thought process/ plan
    answer_as_letter: str = ""
    answer_as_number: float | None = None
    answer_as_boolean: bool | None = None
    answer_as_text: str = ""
    extracted_variables: List[str] = field(default_factory=list)
    steps_for_answer:  List[str] = field(default_factory=list)

