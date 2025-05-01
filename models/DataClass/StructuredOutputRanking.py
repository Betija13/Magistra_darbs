from pydantic import BaseModel, Field, validator
from dataclasses import field
from pydantic.dataclasses import dataclass
from typing import List
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Rankings:
    answer_idx: int
    answer_score: float


class OutputRankingScores(BaseModel):
    chain_of_thought: str = Field(description="Explain your reasoning step-by-step.")
    answer_scores: list[Rankings] = Field(description="The scores for each answer from 0 to 10.")


class OutputRankingBestIdx(BaseModel):
    chain_of_thought: str = Field(description="Explain your reasoning step-by-step.")
    best_idx: int = Field(description="The index of the best answer.")
    best_answer_score: float = Field(description="The score of the best answer from 0 to 10.")


@dataclass_json
@dataclass
class StructuredOutputRanking:
    solution_explanation: str = ""  # thought process/ plan
    best_idx: int | None = None
    best_answer: str | None = None
    best_answer_score: float | None = None
    ranking_scores: List[float] = field(default_factory=list)  # scores for each answer
    answers_sorted: List[str] = field(default_factory=list)
    answer_rankings: List[Rankings] = field(default_factory=list)

