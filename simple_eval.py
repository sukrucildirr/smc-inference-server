import lm_eval
from evaluation.llamppl_inference_adapter import LLaMPPLInferenceModel
import json
import os

# --- Configuration and Initialization ---

# Initialize the custom language model for evaluation.
# This model uses a probabilistic programming approach with specified parameters.
#   - num_particles: The number of particles to use for SMC inference.
#   - beam_factor: Factor for beam search, influencing the breadth of the search.
#   - num_tokens: The maximum number of tokens to generate.
lm_obj = LLaMPPLInferenceModel(num_particles=3, beam_factor=1, num_tokens=50)

# Initialize the TaskManager for lm-evaluation-harness.
# It's crucial to provide the correct path to where the evaluation tasks are defined.
# This allows the evaluator to load the specified tasks for benchmarking.
# NOTE: Replace "/path/to/lm-evaluation-harness/lm_eval/tasks/" with the actual path
#       to your `lm-evaluation-harness` tasks directory.
task_manager = lm_eval.tasks.TaskManager(include_path="/path/to/lm-evaluation-harness/lm_eval/tasks/")

# --- Model Evaluation ---

# Perform the evaluation using lm-evaluation-harness's simple_evaluate function.
# This function runs the specified language model against a set of tasks.
results = lm_eval.simple_evaluate(
    model=lm_obj,                  # The language model object to evaluate.
    tasks=["mmlu_smc_regex"],      # A list of tasks to run. "mmlu_smc_regex" is an example task.
    num_fewshot=0,                 # Number of few-shot examples to provide to the model (0 for zero-shot).
    batch_size=1,                  # Number of examples to process in a single batch.
    task_manager=task_manager,     # The initialized TaskManager.
    limit=1                        # Optional: Limit the number of examples evaluated per task (useful for testing).
)

# --- Results Handling ---

# Define the directory and filename for storing results.
output_dir = "results"
output_filename = os.path.join(output_dir, "run_data.json")

# Ensure the output directory exists. If not, create it.
os.makedirs(output_dir, exist_ok=True)

# Save the evaluation results to a JSON file.
# The 'indent=4' argument makes the JSON output human-readable.
try:
    with open(output_filename, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Evaluation results successfully saved to {output_filename}")
except IOError as e:
    print(f"Error saving results to file {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred while saving results: {e}")