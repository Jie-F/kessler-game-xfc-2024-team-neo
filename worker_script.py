# worker_script.py
import concurrent.futures
import time

def cpu_intensive_task(number):
    print(f"Processing {number}")
    time.sleep(2)  # Simulating a CPU-intensive task
    result = number * number
    print(f"Processed {number}, Result: {result}")
    return result

class Worker:
    def run_tasks(self):
        numbers = [1, 2, 3, 4, 5]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(cpu_intensive_task, numbers)
            for result in results:
                print(result)

# The calling script remains the same as before.
