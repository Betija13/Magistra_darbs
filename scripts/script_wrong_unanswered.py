import csv
import os

info_results_file = '../datasets/info_results.csv'
dataset_folder = '../datasets'

# Step 1: Read the info_results.csv file
with open(info_results_file, 'r', encoding='utf-8') as info_file:
    reader = csv.DictReader(info_file)
    # Step 2: Filter rows with dataset_name 'AQuA-RAT'
    filtered_rows = [row for row in reader if row['dataset_name'] == 'AQuA-RAT' and int(row['id']) > 43]
all_incorrect = {}
# Step 3: Process each result_file_name
for row in filtered_rows:
    result_file_name = row['result_file_name']
    # Extract the part after '/datasets/'
    relative_path = result_file_name.split('/datasets/')[-1]
    result_file_path = os.path.join(dataset_folder, relative_path)

    # Step 4: Read the corresponding result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        result_reader = csv.DictReader(result_file)
        # Step 5: Filter rows with correct == 'false'
        incorrect_rows = [result_row for result_row in result_reader if result_row['correct'].lower() == 'false']
        incorrect_row_ids = [incorrect_row['id'] for incorrect_row in incorrect_rows]
        print(f'Incorrect rows in {result_file_path}: {incorrect_row_ids}')
        empty_rows = [result_row['id'] for result_row in result_reader if result_row['llm'].lower() == '']
        print(f'Empty rows in {result_file_path}: {empty_rows}')
        print('--------------------------------------')
        for incorrect_row_id in incorrect_row_ids:
            if all_incorrect.get(incorrect_row_id):
                all_incorrect[incorrect_row_id] += 1
            else:
                all_incorrect[incorrect_row_id] = 1
print(all_incorrect)
sorted_incorrect = dict(sorted(all_incorrect.items(), key=lambda item: item[1], reverse=True))
print(sorted_incorrect)
print(len(filtered_rows))
