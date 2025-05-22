import asyncio
import logging
import math
import numpy as np
import os
import time
import torch
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from llamppl import Model, LMContext, CachedCausalLM, TokenCategorical, Token, smc_steer
from typing import List, Dict, Any
from util.request_model import GenerationRequest # Assuming this defines pydantic models for requests


# --- Configuration ---
# Name of the pre-trained language model to load.
MODEL_NAME = "NousResearch/Hermes-3-Llama-3.2-3B"

# --- Parallel Server Configuration ---
# WORKER_ID is used to differentiate logs and potentially for future distributed logic.
# Each worker (container) will be assigned a single GPU via CUDA_VISIBLE_DEVICES
WORKER_ID = int(os.environ.get("WORKER_ID", 0))
# NUM_GPUS is effectively 1 for each isolated worker process as it sees only its assigned GPU.
# The orchestration script will handle launching multiple workers for multiple physical GPUs.

# --- Logging Setup ---
# Configure basic logging for the application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Model State ---
# List to hold the loaded language model instances. Currently designed for a single model per worker.
lm_models: List[CachedCausalLM] = []
# Index of the currently active model (relevant for future multi-model scenarios, currently 0).
current_model_idx: int = 0
# List of asyncio locks, one for each model, to ensure exclusive access during generation.
model_locks: List[asyncio.Lock] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the FastAPI application, handling model loading at startup
    and cleanup at shutdown. This ensures the model is loaded once and properly
    released from GPU memory.
    """
    global lm_models, model_locks

    local_gpu_id = 0 # Each worker is assumed to be assigned a single GPU via CUDA_VISIBLE_DEVICES
    logger.info(f"Worker {WORKER_ID} initializing model on local GPU {local_gpu_id}")

    # Configuration options for the vLLM engine.
    kwargs = {
        "engine_opts": {
            "gpu_memory_utilization": 0.85, # Proportion of GPU memory to be used by the model.
            "max_model_len": 1024,         # Maximum sequence length that the model can handle.
            "enforce_eager": True          # Enforce eager execution for debugging/profiling.
        }
    }

    logger.info(f"Loading model '{MODEL_NAME}' on local GPU {local_gpu_id}...")
    try:
        # Load the pre-trained causal language model using vLLM backend.
        lm = CachedCausalLM.from_pretrained(
            MODEL_NAME,
            backend='vllm',
            **kwargs
        )
        lm.batch_size = 1 # Set batch size for inference.

        lm_models = [lm] # Store the loaded model.
        model_locks = [asyncio.Lock()] # Create a lock for the model.

        # Model warmup: Perform a dummy generation to ensure the model is fully loaded
        # and optimized before handling actual requests. This reduces first-request latency.
        logger.info(f"Warming up model on local GPU {local_gpu_id}...")
        dummy_model = FixedLengthSentenceModel(lm=lm, prompt="Hello, world", num_tokens=20)
        # Use smc_steer with minimal particles/beam factor for a quick warmup run.
        await smc_steer(dummy_model, 1, 1)
        logger.info(f"Model warmup complete on local GPU {local_gpu_id}")
    except Exception as e:
        logger.error(f"Failed to load model on worker {WORKER_ID}: {e}", exc_info=True)
        # Re-raise the exception to prevent the FastAPI app from starting if model loading fails.
        raise

    yield # Yield control to the FastAPI application to start processing requests.

    # --- Application Shutdown ---
    logger.info(f"Shutting down model on worker {WORKER_ID} on local GPU {local_gpu_id}...")
    # Release model resources.
    for lm_instance in lm_models:
        del lm_instance # Explicitly delete the model instance.

    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear GPU memory cache.

# Initialize FastAPI application with the defined lifespan context.
app = FastAPI(lifespan=lifespan)

# --- SMC Constraint Model Definition ---

class FixedLengthSentenceModel(Model):
    """
    A custom probabilistic model extending `llamppl.Model` to enforce specific
    constraints during text generation using SMC (Sequential Monte Carlo) steering.
    This model generates coherent sentences of a *fixed length* and ensures they
    end with a period.
    """
    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        num_tokens: int = 10,
        temperature: float = 1.0
    ):
        """
        Initializes the FixedLengthSentenceModel.

        Args:
            lm (CachedCausalLM): The underlying language model used for token prediction.
            prompt (str): The initial text prompt for generation.
            num_tokens (int): The exact number of tokens to generate for the sentence.
            temperature (float): Sampling temperature for text generation.
        """
        super().__init__()

        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.num_tokens = num_tokens
        self.generated_tokens: List[Token] = [] # Stores the generated tokens.
        self.max_tokens = num_tokens # Maximum number of tokens to generate (same as num_tokens).
        self.eos_token = lm.tokenizer.eos_token_id # End-of-sentence token ID.

        # Pre-compute token IDs that represent a period.
        self.period_tokens = set()
        if self.lm.vocab:
            for i, token_str in enumerate(self.lm.vocab):
                if token_str and token_str.endswith('.'):
                    self.period_tokens.add(i)
        else:
            logger.warning("LM vocab is empty or None, period_tokens will not be set. This may impact constraints.")

    async def step(self):
        """
        Generates the next token for the sentence, applying length and ending constraints.
        This method is called iteratively by `smc_steer`.
        """
        current_length = len(self.generated_tokens)

        # Condition for finishing generation: exact length reached and ends with a period.
        if current_length >= self.num_tokens:
            self.condition(current_length == self.num_tokens) # Enforce exact length.

            # Check if the last generated token is a period token.
            if self.generated_tokens and self.generated_tokens[-1].token_id in self.period_tokens:
                self.condition(True) # Condition satisfied.
            else:
                self.condition(False) # Condition not satisfied.
            self.finish() # Signal that generation for this particle is complete.
            return

        # Get the probability distribution for the next token from the language model.
        next_dist = self.context.next_token()

        # Apply constraints based on the current length:
        if current_length == self.num_tokens - 1:
            # If it's the last token, force it to be one ending with a period.
            period_mask = self.period_tokens
            if not period_mask:
                logger.error("period_tokens mask is empty for final token constraint. This particle will be discarded.")
                self.condition(False) # Fail the particle if no period tokens are found.
                self.finish()
                return
            # Observe the distribution masked to only period-ending tokens.
            await self.observe(self.context.mask_dist(period_mask), True)
        else:
            # For non-final tokens, prevent period tokens and EOS to ensure the sentence
            # continues until the target length and only ends with a period at the end.
            all_token_ids = set(range(len(self.lm.vocab)))
            non_period_mask = all_token_ids - self.period_tokens - {self.lm.tokenizer.eos_token_id}
            if not non_period_mask:
                logger.error("non_period_mask is empty for non-final token constraint. This particle will be discarded.")
                self.condition(False) # Fail the particle if no valid non-period tokens are found.
                self.finish()
                return
            # Observe the distribution masked to exclude period-ending and EOS tokens.
            await self.observe(self.context.mask_dist(non_period_mask), True)

        # Sample the next token based on the (possibly masked) distribution.
        token = await self.sample(next_dist)
        self.generated_tokens.append(token) # Add the sampled token to the sequence.


# --- Metrics for Server Performance ---
start_time = time.time()       # Timestamp when the server started.
total_requests = 0             # Total number of generation requests processed.
total_tokens = 0               # Total number of tokens generated across all requests.
request_times: List[float] = [] # List to store the duration of each generation request.

# --- API Endpoints and Helper Methods ---

async def get_next_available_model() -> (int, CachedCausalLM, asyncio.Lock):
    """
    Returns the current model and its lock. Designed for extensibility to
    support multiple models or more complex load balancing if needed in the future.
    Currently, always returns the first (and only) model.

    Returns:
        tuple: A tuple containing the model index, the CachedCausalLM instance, and its asyncio.Lock.
    """
    # In a single-model-per-worker setup, we just return the first model.
    if not lm_models:
        raise RuntimeError("Language model not initialized.")
    return 0, lm_models[0], model_locks[0]

async def generate_text(request: GenerationRequest) -> Dict[str, Any]:
    """
    Handles the core text generation logic using SMC steering with the defined constraints.

    Args:
        request (GenerationRequest): An object containing generation parameters like prompt,
                                     number of particles, beam factor, and number of tokens.

    Returns:
        Dict[str, Any]: A dictionary containing the generated text.

    Raises:
        ValueError: If `num_particles` or `beam_factor` are invalid.
        RuntimeError: If the language model is not initialized.
        HTTPException: For errors during the SMC steering process.
    """
    if request.num_particles <= 0 or request.beam_factor <= 0:
        raise ValueError("num_particles and beam_factor must be positive integers.")

    if not lm_models:
        logger.error("Model not initialized. Cannot generate text.")
        raise RuntimeError("Model not initialized.")

    # Get the model instance and its lock.
    model_idx, lm, lock = await get_next_available_model()

    # Acquire the lock to ensure exclusive access to the model during generation.
    async with lock:
        # Inference mode disables gradient calculations, saving memory and speeding up inference.
        with torch.inference_mode():
            # Instantiate the custom probabilistic model with request parameters.
            model = FixedLengthSentenceModel(
                lm=lm,
                prompt=request.prompt,
                num_tokens=request.num_tokens,
                temperature=request.temperature
            )

            try:
                # Perform Sequential Monte Carlo (SMC) steering.
                particles = await smc_steer(
                    model,
                    request.num_particles,
                    request.beam_factor
                )

                if not particles:
                    logger.warning(f"SMC steering returned no valid particles for request: {request.prompt}")
                    return {"generated_text": "Generation failed: No valid particles found."}

                # Select the best particle based on its weight.
                best_particle = max(particles, key=lambda p: p.weight)
                generated_text = str(best_particle.context) # Extract the generated text.
                return {"generated_text": generated_text}
            except Exception as e:
                logger.error(f"Error during SMC steering for worker {WORKER_ID}: {e}", exc_info=True)
                raise # Re-raise to be caught by the API endpoint handler.

@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Provides a health check endpoint for the server.

    Returns:
        Dict[str, Any]: A dictionary indicating the server's status, worker ID,
                        and local GPU ID.
    """
    if not lm_models:
        return {"status": "loading"} # Model is still being initialized.
    return {
        "status": "ready",
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0 # Assuming one GPU per worker.
    }

@app.get("/model_info")
async def model_info() -> Dict[str, Any]:
    """
    Returns information about the loaded language model.

    Returns:
        Dict[str, Any]: A dictionary containing model details like EOT token ID,
                        max sequence length, worker ID, and local GPU ID.
    """
    if not lm_models:
        return {"status": "loading"}

    # Safely access tokenizer attributes.
    eos_token_id = lm_models[0].tokenizer.eos_token_id if hasattr(lm_models[0].tokenizer, 'eos_token_id') else None
    max_model_len = lm_models[0].max_model_len if hasattr(lm_models[0], 'max_model_len') else None

    return {
        "eot_token_id": eos_token_id,
        "max_length": max_model_len,
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0
    }

@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Provides real-time statistics about the server's performance.

    Returns:
        Dict[str, Any]: A dictionary containing total requests, total tokens generated,
                        tokens per second, average latency, uptime, worker ID, and GPU ID.
    """
    global start_time, total_requests, total_tokens, request_times

    elapsed = time.time() - start_time
    # Calculate average latency, avoiding division by zero.
    avg_latency = sum(request_times) / len(request_times) if request_times else 0

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
        "avg_latency": avg_latency,
        "uptime_seconds": elapsed,
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0
    }

@app.post("/generate")
async def generate(request: GenerationRequest) -> Dict[str, Any]:
    """
    Primary endpoint for text generation requests. It processes the request,
    calls the generation logic, and updates performance metrics.

    Args:
        request (GenerationRequest): The incoming request payload conforming to `GenerationRequest` model.

    Returns:
        Dict[str, Any]: The generated text from the model or an error message.
    """
    global total_requests, total_tokens, request_times

    start_time_req = time.time()
    try:
        # Call the core generation logic.
        result = await generate_text(request)
        duration = time.time() - start_time_req

        request_times.append(duration)
        total_requests += 1
        # It's better to use the tokenizer to count tokens accurately,
        # especially for models where character length doesn't directly map to token length.
        # Assuming lm_models[0] is available and has a tokenizer.
        if lm_models and hasattr(lm_models[0], 'tokenizer') and hasattr(lm_models[0].tokenizer, 'encode'):
            total_tokens += len(lm_models[0].tokenizer.encode(result.get("generated_text", "")))
        else:
            # Fallback to character count if tokenizer isn't readily available, but log a warning.
            total_tokens += len(result.get("generated_text", ""))
            logger.warning("Tokenizer not available for accurate token count. Falling back to character count.")


        # Log periodic statistics to monitor performance.
        if total_requests % 10 == 0:
            elapsed = time.time() - start_time
            # Calculate rolling average latency for the last 100 requests.
            avg_latency = sum(request_times[-100:]) / min(100, len(request_times))
            logger.info(f"Worker {WORKER_ID} | GPU (local) {0} | Requests: {total_requests} | Avg latency (last {min(100, len(request_times))} req): {avg_latency:.4f}s")

        return result
    except ValueError as e:
        logger.error(f"API Error (Bad Request) in /generate for worker {WORKER_ID}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"API Error (Server Misconfiguration) in /generate for worker {WORKER_ID}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server not ready or misconfigured. Please try again later.")
    except Exception as e:
        logger.error(f"API Error (Internal Server Error) in /generate for worker {WORKER_ID}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during generation: {str(e)}")

# --- Main Execution ---
if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn.
    uvicorn.run(
        app,
        host="0.0.0.0", # Listen on all available network interfaces.
        port=8001,      # The port this specific worker listens on.
        workers=1,      # Each Uvicorn process is a worker; we want 1 worker per GPU/container.
        # Note: 'reload=True' is generally for development and should be false in production.
    )