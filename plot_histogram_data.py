import matplotlib.pyplot as plt
import pandas as pd

def plot_histograms_from_file(file_path):
    """
    Reads a CSV file and plots histograms for each column.
    The first line of the file is expected to contain comma-separated column names.

    :param file_path: Path of the CSV file.
    """
    # Read the file into a DataFrame
    data = pd.read_csv(file_path)

    # Create a histogram for each column
    for column in data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(data[column], bins=100, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# Path to the file (modify this path as needed)
file_path = 'Safe Maneuvers.txt'

# Call the function to plot histograms
plot_histograms_from_file(file_path)
