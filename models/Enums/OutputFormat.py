from enum import Enum


class OutputFormat(Enum):
    STRUCTURED_COT = 'STRUCTURED_COT'  # answer field + cot field
    STRUCTURED_ANSWER = 'STRUCTURED_ANSWER'  # only 1 answer field
    STRUCTURED_EXTRA = 'STRUCTURED_EXTRA'   # answer, cot and other extra fields
    NO_FORMAT = 'NO_FORMAT'  # regular llm output without any structured response format
