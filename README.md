# Magistra_darbs

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
* MODEL_NAME - name of used llm
* PREDIFINED_DATASETS - list of datasets that ill be used (if none, will use all datasets)
* PREDIFINED_FILES - list of files that will be used (if none, will use all files in the dataset)
* USE_SYSTEM_PROMPT_STRUCTURE - boolean if use system prompt structure as described in https://medium.com/@niall.mcnulty/writing-an-o1-prompt-that-works-16ee921b5859

## Results
All main info about results can be found in datasets/info_results.csv

Detailed results about each experiment can be found in datasets/<dataset_name>/results/<file_name>

File name structure is as follows `<method name>_<date>_<dataset file name>_<custom name>.csv`


