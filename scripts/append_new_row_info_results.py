import csv
import os
new_field_name = 'llm_model'
default_new_value_name = 'gpt-4o-mini'
# Define the folder and file name
folder = '../datasets'
filename = 'info_results.csv'
file_path = os.path.join(folder, filename)

# Read the existing CSV file
with open(file_path, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Add the new column 'reward_method' with a default value of None
if new_field_name not in fieldnames:
    fieldnames.append(new_field_name)
    for row in rows:
        row[new_field_name] = default_new_value_name

# Write the updated data back to the CSV file
with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)