from enum import Enum


class Datasets(str, Enum):
    AQUA = 'AQuA-RAT'
    LAST_LETTER = 'LastLetterConcat'
    MATHQA = 'MATHQA'
    MMLU = 'MMLU'
    RIDDLESENSE = 'RiddleSense'
    THEOREMAQA = 'TheoremaQA'
