from enum import Enum


class Method(str, Enum):
    A_1 = 'A_ZERO_SHOT'
    A_2 = 'A_N_SAMPLING'
    MUT = 'MUTATION'
    MUT_C = 'MUTATION_CORRECT'  # As next examples are given those prompts that produced a correct answer
    STRUCT_MUT_C = 'STRUCTURED_MUTATION_CORRECT'
    MUT_M = 'MUTATION_MAJOR'  # As next examples are given those prompts that produced major answer
    STRUCT_MUT_M = 'STRUCTURED_MUTATION_MAJOR'
    MUT_E = 'MUTATION_EDIT'  # Mutation is ongoing until the correct answer is produced or the limit is reached
    STRUCT = 'STRUCTURED_OUTPUT' # Structured output with explanation
    STRUCT_ANS = 'STRUCTURED_ONLY_ANSWER' #Structured output with only answer
    PS = 'PLAN_AND_SOLVE'
    PS_PLUS = 'PLAN_AND_SOLVE_PLUS'
    ZS_COT = 'ZS_COT'
    TWO_PROMPTS = 'TWO_PROMPTS'
