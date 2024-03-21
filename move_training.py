# 2024-03-21
# Move the training data from the local directory to a common target directory

import os
import shutil
import time

# Paths to your directories
source_dir = r"C:\Users\Jie Fan\Documents\GitHub\kessler-game-xfc-2024-team-neo\training_v3"
target_dir = r"A:\Programming\XFC 2024\Neo Training\training_v3"

def move_files(src, dst):
    """
    Attempts to move files from the source directory to the destination.
    Moves a file only if it hasn't been modified in the last minute.
    """
    try:
        for filename in os.listdir(src):
            source_path = os.path.join(src, filename)
            destination_path = os.path.join(dst, filename)
            
            # Get the last modification time and the current time
            last_mod_time = os.path.getmtime(source_path)
            current_time = time.time()
            
            # Check if the file was last modified at least 60 seconds ago
            if current_time - last_mod_time > 60:
                try:
                    # Attempt to move the file
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {filename}")
                except Exception as e:
                    # If an error occurs during file move, print the error
                    print(f"Error moving {filename}: {e}")
            else:
                print(f"Skipped recently modified file: {filename}")
    except Exception as e:
        # If an error occurs accessing the source directory, print the error
        print(f"Error accessing source directory: {e}")

# Example of a continuous loop that checks every 60 seconds
while True:
    move_files(source_dir, target_dir)
    # Wait for a specified time before checking again
    time.sleep(60)  # Adjust the sleep time as needed
