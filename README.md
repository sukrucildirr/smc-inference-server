# smc-inference-server

Run [Sequential Monte Carlo Steering](https://arxiv.org/abs/2306.03081) as a local inference server for exploration, demos, and evaluation with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)!

See our [technical blog publication](https://smc-blogpost.vercel.app/) for more information.

## Overview

This repository is designed to be a simple and extensible starting point for hosting models that utilize the [llamppl](https://github.com/genlm/llamppl) framework for steering. While `llampl` has a vLLM backend, it's not set up out-of-the-box to run benchmarks, so the first step was to wrap a simple API endpoint around the framework.

Along with this, we share a simple distributed framework for spawning multiple SMC backends across your GPUs to maximize request processing, making it possible to run SMC benchmarks on the full MMLU set (and others!) in a matter of hours (WORK IN PROGRESS TBD).

### Dependencies

```
pip install torch vllm llamppl fastapi uvicorn
```

### Usage

To test SMC framework works as expected:
```
python examples/entropy.py
```

To start the SMC Inference Server:

```
uvicorn server:app --host 0.0.0.0
```

Requests can be made to the endpoint from the command line like so:

```
curl -X POST "http://0.0.0.0:8000/generate"      -H "Content-Type: application/json"      -d '{
           "prompt": "Tell me a joke",
           "num_particles": 3,
           "beam_factor": 1,
           "num_tokens": 20
         }'
```

### Constraints

To create new constraints for SMC, edit the `server.py` file, where the `FixedLengthSentenceModel` is defined. Examples of constraints can be found in the `examples` directory, covering the constrainted we explored in our blog post.

The request body in `util/request_model.py` can also be edited to include more parameters relevant to your constraint design. Keep in mind the request body for `generate_until` inside `evaluation/tasks/llamppl_inference_adapter.py` will need to be updated as well if you plan to run benchmarks.

## Evaluation Steps

This evaluation framework relies on `generate_until` calls from our custom `LLaMPPLInferenceAdapter`, and uses regex to extract the answer from the response. The sample task definition in `evaluation/tasks/mmlu_smc_regex.yaml` covers how this is set up.

### LM Eval Harness

Clone the lm-evaluation-harness repository following the steps [there](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

To run MMLU with SMC, copy the sample task definition `evaluation/tasks/mmlu_smc_regex.yaml` to the `tasks/llama3/instruct/mmlu/` directory of `lm-evaluation-harness` (/path/to/lm-evaluation-harness/lm_eval/tasks/llama3/instruct/mmlu/ is the format). 

All future task definitions for generative MMLU tasks should go in here as well. Other task definitions can be placed in their respective directories.

Lastly, add the path to the above `tasks` directory to `simple_eval.py`, so `lm-evaluation-harness` can find your new task definition.

Run the server, and in another terminal, run `python simple_eval.py` to run the evaluation. Results will be saved to `run_data.json`.

## TODO

Parallelization is next on our roadmap, with the move to vLLM we have some bugs around standing up parallel model instances on separate GPUs. This wasn't an issue in the past, but right now we just have a single instance stood up on a single GPU for a starting point. There will be updates to this soon.