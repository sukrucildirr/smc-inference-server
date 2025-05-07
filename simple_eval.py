import lm_eval
from evaluation.llamppl_inference_adapter import LLaMPPLInferenceModel

import json

lm_obj = LLaMPPLInferenceModel(num_particles=3, beam_factor=1, num_tokens=50)

task_manager = lm_eval.tasks.TaskManager(include_path="/path/to/lm-evaluation-harness/lm_eval/tasks/")

results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["mmlu_smc_regex"],
            num_fewshot=0,
            batch_size=1,
            task_manager=task_manager,
            limit=1
)

#print(results)

with open("results/run_data.json", "w") as file:
        json.dump(results, file, indent=4)
        file.close()