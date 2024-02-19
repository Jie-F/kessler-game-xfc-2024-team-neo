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

# Initialize an empty list to keep track of top 3 scores
top_scores = []

for run in all_data:
    current_score = run['team_1_hits']
    current_chromosome = run['chromosome']
    # Append the current run's score and chromosome to the list
    top_scores.append((current_score, current_chromosome))

# Sort the list by score in descending order and keep only the top 3
top_scores = sorted(top_scores, key=lambda x: x[0], reverse=True)[:5]

# Print the top 3 scores and their chromosomes
for i, (score, chromosome) in enumerate(top_scores, start=1):
    print(f"Top {i} score is {score} with chromosome {chromosome}")
