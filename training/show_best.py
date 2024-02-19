import os
import json

def read_and_process_json_files(directory="."):
    all_data = []
    # Iterate through all files in the current directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # Check if the file is a JSON file
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf8') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):  # Ensure the JSON content is a list
                        all_data.extend(data)
                    else:
                        print(f"File {filename} does not contain a list.")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file {filename}.")
    return all_data

# Call the function to start processing
all_data = read_and_process_json_files()

best_score = 0
best_chromosome = None
for run in all_data:
    if run['team_1_hits'] > best_score:
        best_score = run['team_1_hits']
        best_chromosome = run['chromosome']

print(f"The best score is {best_score} with chromosome {best_chromosome}")
