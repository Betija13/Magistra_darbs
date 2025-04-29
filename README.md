# Maģistra darba kods

## Setup

all necessary packages are listed in requirements.txt

!Note: when installing torch, go to https://pytorch.org/get-started/locally/ and select the suitable version for yourself 


## Running experiments
To run experiments, run file scripts/script_answers.py

In the file, uncomment the code under the mentioned method or change parameters and run as it is.

At the beginning of the file, you can set the parameters for the experiment.

* CUSTOM_NAME - custom name will be added to output file name
* TOTAL_COUNT - maximum count of questions (if smaller than dataset size)
* TEMPERATURE - temperature of model
* ANSWER_COUNT - count of answers
* N_SAMPLES_MUT - count of mutations if method is mutation method
* METHOD - used method
* METHOD_NAME_FILE - name of the method (METHOD by default)
* REWARD_METHOD - used reward method, if applicable
* MODEL_NAME - name of used OpenAI llm
* PREDEFINED_DATASETS - list of datasets that ill be used (if none, will use all datasets)
* PREDEFINED_FILES - list of files that will be used (if none, will use all files in the dataset)
* USE_SYSTEM_PROMPT_STRUCTURE - boolean if use system prompt structure as described in https://medium.com/@niall.mcnulty/writing-an-o1-prompt-that-works-16ee921b5859
* MUTATION_UNTIL_SATISFIED - if mutation is continuing until some satisfactory value is reached (e.g. correct answer)
* OUTPUT_FORMAT - output format from llm. If NO_FORMAT then output is regular str, otherwise it is structured.
* REWARD_NAME - Name of reward model used

## Results
All main info about results can be found in datasets/info_results.csv

Detailed results about each experiment can be found in datasets/<dataset_name>/results/<file_name>

File name structure is as follows 

(older version) `<method name>_<date>_<dataset file name>_<custom name>.csv`

(from 29/04/2025) `<method name>_<[opt]CONT-US>_<[opt]reward method>_O-<Output format>_<I-N\I-S>_<date>_<dataset file name>_<custom name>.csv`

* [optional] CONT-US is MUTATION_UNTIL_SATISFIED.
* [optional] reward method is REWARD_METHOD if any is chosen
* I-N\I-S: input format. S - structure, N - no structure (USE_SYSTEM_PROMPT_STRUCTURE)

