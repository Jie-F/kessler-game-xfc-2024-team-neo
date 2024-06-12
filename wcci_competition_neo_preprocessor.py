# Go through "neo_controller.py" and remove lines that contain "REMOVE_FOR_COMPETITION".
# This script will read from "neo_controller.py", filter out the unwanted lines, and write the result to a new file.

# Define the source file path and the destination file path
source_file_path = 'neo_controller.py'
destination_file_path = r'WCCI SUBMISSIONS\neo_controller.py'

# Define the pattern to search for lines to remove
pattern_to_remove = 'REMOVE_FOR_COMPETITION'

# Open the source file, read lines, and filter out the unwanted lines
with open(source_file_path, 'r', encoding='utf8') as source_file:
    lines = source_file.readlines()
print('Read in the input file')
# Writing to the destination file, excluding lines containing the specified pattern
with open(destination_file_path, 'w', encoding='utf8') as dest_file:
    for line in lines:
        if pattern_to_remove not in line:
            dest_file.write(line)

print('Done processing!')
