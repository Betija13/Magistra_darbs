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
    E = "E"


class StructuredOutputModelMultipleChoice(BaseModel):
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
    answer_as_number: float = 0.0
    answer_as_boolean: bool = False
    answer_as_text: str = ""

