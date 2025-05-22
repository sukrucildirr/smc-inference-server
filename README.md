# 🚀 SMC Inference Server

Run [Sequential Monte Carlo Steering](https://arxiv.org/abs/2306.03081) as a robust local inference server. This project is designed for exploration, demos, and efficient evaluation with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Dive deeper into the technical aspects with our [technical blog publication](https://smc-blogpost.vercel.app/).

---

## 🌟 Overview

This repository provides a streamlined and extensible foundation for hosting language models that leverage the [llamppl](https://github.com/genlm/llamppl) framework for steerable generation. While `llamppl` integrates with vLLM, this project wraps it with a simple, production-ready API endpoint, making it easier to integrate into your workflows and run benchmarks.

It also includes a distributed framework designed to launch multiple SMC backends across your GPUs, maximizing request processing throughput. This capability aims to enable running SMC benchmarks on extensive datasets like MMLU in a matter of hours.

---

## 🛠️ Setup and Installation

Follow these steps to get the SMC Inference Server up and running.

### Prerequisites

* **Docker**: Ensure Docker is installed and running on your system.
* **CUDA**: Your system should have NVIDIA GPUs and CUDA drivers installed for `vLLM` to function correctly.
* **Python Dependencies (for development/local setup without Docker)**:
    ```bash
    pip install torch vllm llamppl fastapi uvicorn httpx numpy
    ```

### Step-by-Step Deployment

We recommend using Docker for a consistent and isolated environment.

1.  **Build the Docker Image**:
    First, build the Docker image for your SMC Inference Server. This command compiles the necessary dependencies and sets up your environment.

    ```bash
    docker build -t llamppl-inference-server .
    ```

2.  **Start Backend Inference Servers**:
    Each backend server will run in its own Docker container, utilizing a dedicated GPU. The `./start_servers.sh` script handles this orchestration. Make sure your `CUDA_VISIBLE_DEVICES` environment variable is correctly configured if you want to assign specific GPUs.

    ```bash
    ./start_servers.sh
    ```
    * **Note**: This script expects your system to have available GPUs and will launch a server for each `WORKER_ID` as defined in the script. Modify the script if you need to control the number of workers or GPU assignments.

3.  **Start the Load Balancer**:
    In a **separate terminal**, start the FastAPI load balancer. This component will distribute incoming requests among your backend inference servers.

    ```bash
    uvicorn load_balancer:app --host 0.0.0.0 --port 8000
    ```
    * The load balancer will be accessible at `http://0.0.0.0:8000`.

---

## 🚀 Usage

Once the servers are running, you can start sending generation requests.

### Testing the SMC Framework

To verify that the `llamppl` framework is working as expected with a simple example:

```bash
python examples/entropy.py
```

## 📝 Evaluation

This evaluation framework relies on `generate_until` calls from our custom `LLaMPPLInferenceAdapter`, and uses regex to extract the answer from the response. The sample task definition in `evaluation/tasks/mmlu_smc_regex.yaml` covers how this is set up.

### LM Eval Harness

Clone the lm-evaluation-harness repository following the steps [there](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

To run MMLU with SMC, copy the sample task definition `evaluation/tasks/mmlu_smc_regex.yaml` to the `tasks/llama3/instruct/mmlu/` directory of `lm-evaluation-harness` (/path/to/lm-evaluation-harness/lm_eval/tasks/llama3/instruct/mmlu/ is the format). 

All future task definitions for generative MMLU tasks should go in here as well. Other task definitions can be placed in their respective directories.

Lastly, add the path to the above `tasks` directory to `simple_eval.py`, so `lm-evaluation-harness` can find your new task definition.

Run the server + load balancer, and in another terminal, run `python simple_eval.py` to run the evaluation. Results will be saved to `run_data.json`.