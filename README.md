# Maģistra darba kods

## Setup

all necessary packages are listed in requirements.txt

!Note: when installing torch, go to https://pytorch.org/get-started/locally/ and select the suitable version for yourself 


## Running experiments

Run in terminal `python scripts/script_answers.py --TASK_VALUE <task_value>` choosing which task to run. All task names can be found under models/Enums/Tasks.py

To iterate through prompts add `--PROMPT_LIST_EXISTING` choosing one of prompt_names_list values from constants.py or add `--PROMPTS_ITERATION` manually adding prompts.

Also van change these params:

* CUSTOM_NAME - custom name will be added to output file name
* TOTAL_COUNT - maximum count of questions (if smaller than dataset size)
* TEMPERATURE - temperature of model
* ANSWER_COUNT - count of answers
* N_SAMPLES_MUT - count of mutations if method is mutation method
* METHOD - used method. Choose in args by setting `--METHOD_VALUE` to one of the values from models/Enums/Method.py
* METHOD_NAME_FILE - name of the method that will be saved in filename (METHOD by default)
* REWARD_METHOD - used reward method, if applicable. Choose in args by setting `--REWARD_METHOD_VALUE` to one of the values from models/Enums/RewardMethod.py
* REWARD_NAME - Name of reward model used. Choose in args by setting `--REWARD_NAME_VALUE` to one of the values from models/Enums/RewardModelNames.py
* MODEL_NAME - name of used OpenAI llm. Default is 'gpt-4o'
* PREDEFINED_DATASETS - list of datasets that will be used (if none, will use all datasets). In args add to `--DATASETS` adding values from models/Enums/Datasets.py
* PREDEFINED_FILES - list of files that will be used (if none, will use all files in the dataset)
* USE_SYSTEM_PROMPT_STRUCTURE - boolean if use system prompt structure as described in https://medium.com/@niall.mcnulty/writing-an-o1-prompt-that-works-16ee921b5859
* MUTATION_UNTIL_SATISFIED - if mutation is continuing until some satisfactory value is reached (e.g. correct answer)
* OUTPUT_FORMAT - output format from llm. If NO_FORMAT then output is regular str, otherwise it is structured. Choose in args by setting `--OUTPUT_FORMAT` to one of values from models/Enums/OutputFormat.py
* SAME_START - For mutation methods, whether to use the same first prompt for all mutations. True = Mutation always start from the same point.
* USE_EXAMPLE_MUT - For mutation methods, whether to use Q and A example in mutations.
* GET_DYNAMIC_SCORES_Q_T - When doing mutation, get scores for each answer and mutated task prompt. Save scores in file.
* MUTATE_MUT - For mutation. When getting task prompts, if True, mutates task prompt mutations N times, false, gets N answers from first task prompt mutation.
* ORIGINAL_FILE - (Only for tasks N_SAMPLE_DIFF_SCORE_RM and N_SAMPLE_DIFF_SCORE_RM) Existing Static mutation N answers file.

## Results
All main info about results can be found in datasets/info_results.csv

Detailed results about each experiment can be found in datasets/<dataset_name>/results/<file_name>

File name structure is as follows 

(older version) `<method name>_<date>_<dataset file name>_<custom name>.csv`

(from 29/04/2025) `<method name>_<[opt]CONT-US>_<[opt]reward method>_O-<Output format>_<I-N\I-S>_<date>_F-<dataset file name>_<custom name>.csv`

* [optional] CONT-US is MUTATION_UNTIL_SATISFIED.
* [optional] reward method is REWARD_METHOD if any is chosen
* I-N\I-S: input format. S - structure, N - no structure (USE_SYSTEM_PROMPT_STRUCTURE)

