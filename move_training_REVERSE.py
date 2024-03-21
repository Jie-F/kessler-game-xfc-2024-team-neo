# 2024-03-21
# Move the training data from the local directory to a common target directory

import os
import shutil
import time
import json
import hashlib

# Paths to your directories
target_dir = r"C:\Users\Jie Fan\Documents\GitHub\kessler-game-xfc-2024-team-neo\training_v3"
source_dir = r"A:\Programming\XFC 2024\Neo Training\training_v3"

# Threshold for your numeric field in the JSON file
threshold_value = 50300

def hash_file(filepath) -> str:
    """Generate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def file_meets_condition(filepath) -> bool:
    """Check if the file's JSON content has a numeric field below the threshold."""
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Replace 'your_numeric_field' with the actual field name
            return data.get('team_1_total_asteroids_hit', float('inf')) >= threshold_value
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")
        return False  # or handle appropriately
    except Exception as e:
        print(f"Error reading file: {filepath}, {e}")
        return False

def is_file_stable(filepath) -> bool:
    """Check if the file was last modified at least 1 minute ago."""
    last_mod_time = os.path.getmtime(filepath)
    current_time = time.time()
    return (current_time - last_mod_time) > 60.0

def move_files(src, dst) -> None:
    """Move files based on the JSON condition and file hash comparison."""
    try:
        for filename in os.listdir(src):
            source_path = os.path.join(src, filename)
            destination_path = os.path.join(dst, filename)

            if not is_file_stable(source_path):
                print(f"Skipped {filename}: file has been recently modified.")
                continue

            if not file_meets_condition(source_path):
                print(f"Skipped {filename}: the score's really good so we're keeping this one.")
                continue

            # If destination file exists, compare hashes
            if os.path.exists(destination_path):
                source_hash = hash_file(source_path)
                destination_hash = hash_file(destination_path)
                if source_hash == destination_hash:
                    print(f"Destination for {filename} already exists, and it matches the source file. Deleting the source file.")
                    os.unlink(source_path)
                else:
                    print(f"Error: File hashes do not match for {filename}. File not moved.")
                continue

            # Attempt to move the file
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved: {filename}")
            except Exception as e:
                print(f"Error moving {filename}: {e}")
    except Exception as e:
        print(f"Error accessing source directory: {e}")

# Example of a continuous loop that checks every 60 seconds
while True:
    move_files(source_dir, target_dir)
    # Wait for a specified time before checking again
    time.sleep(60.0)  # Adjust the sleep time as needed
