# caller.py
import worker_script

# Instantiate and use the Worker class directly, without the __main__ guard.
worker = worker_script.Worker()
worker.run_tasks()
