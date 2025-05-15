import asyncio
import logging
import math
import numpy as np
import os
import time
import torch
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from llamppl import Model, LMContext, CachedCausalLM, TokenCategorical, Token, smc_steer
from typing import List, Dict, Any
from util.request_model import GenerationRequest # Assuming util/request_model.py exists and defines GenerationRequest


MODEL_NAME = "NousResearch/Hermes-3-Llama-3.2-3B"

########### PARALLEL SERVER CONFIG ##############
# WORKER_ID is used to differentiate logs and potentially for future distributed logic
# Each worker (container) will be assigned a single GPU via CUDA_VISIBLE_DEVICES
WORKER_ID = int(os.environ.get("WORKER_ID", 0))
# NUM_GPUS is effectively 1 for each isolated worker process as it sees only its assigned GPU
# The orchestration script will handle launching multiple workers for multiple physical GPUs.
#################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lm_models = []
current_model_idx = 0
model_locks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lm_models, model_locks

    # Inside this container, only one GPU (index 0) is visible via CUDA_VISIBLE_DEVICES
    # The actual physical GPU ID is managed by the Docker runtime.
    local_gpu_id = 0 
    logger.info(f"Worker {WORKER_ID} initializing model on local GPU {local_gpu_id}")

    # Startup and initialization steps
    kwargs = (
        {"engine_opts": {"gpu_memory_utilization": 0.95, "max_model_len": 1024}}
    )
    
    logger.info(f"Loading model on local GPU {local_gpu_id}...")
    try:
        lm = CachedCausalLM.from_pretrained(
            MODEL_NAME,
            backend='vllm',
            **kwargs
        )
        lm.batch_size = 1 # Ensuring batch size is 1 for individual requests as per current design
        
        lm_models = [lm]
        model_locks = [asyncio.Lock()]

        # Model warmup
        logger.info(f"Warming up model on local GPU {local_gpu_id}...")
        dummy_model = FixedLengthSentenceModel(lm=lm, prompt="Hello, world", num_tokens=20)
        # Using a small number of particles/beam factor for warmup to be quick
        await smc_steer(dummy_model, 1, 1) 
        logger.info(f"Model warmup complete on local GPU {local_gpu_id}")
    except Exception as e:
        logger.error(f"Failed to load model on worker {WORKER_ID}: {e}")
        # Optionally, re-raise the exception to prevent the app from starting
        raise

    yield

    logger.info(f"Shutting down model on worker {WORKER_ID} on local GPU {local_gpu_id}...")
    for lm in lm_models:
        del lm
    # Attempt to clear CUDA memory if necessary (though vLLM usually manages this)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

######## SMC CONSTRAINT MODEL ########

class FixedLengthSentenceModel(Model):
    """
    This FixedLengthSentenceModel demonstrates an example 
    constraint to be used with the SMC Inference Server.
    The constraint here generates coherent sentences of a fixed length.
    """
    def __init__(
        self,
        lm: CachedCausalLM,
        prompt: str,
        num_tokens: int = 10,
        temperature: float = 1.0
    ):
        super().__init__()

        self.lm = lm
        self.context = LMContext(lm, prompt, temperature)
        self.num_tokens = num_tokens
        self.generated_tokens = []
        self.max_tokens = num_tokens
        self.eos_token = lm.tokenizer.eos_token_id

        # Find token IDs that end with a period
        self.period_tokens = set()
        # Ensure vocab is not None and is iterable
        if self.lm.vocab:
            for i, token in enumerate(self.lm.vocab):
                if token and token.endswith('.'): # Check if token is not None
                    self.period_tokens.add(i)
        else:
            logger.warning("LM vocab is empty or None, period_tokens will not be set.")

    async def step(self):
        """Generate exactly num_tokens tokens with a coherent sentence ending in period."""
        current_length = len(self.generated_tokens)
        
        # Condition on exact length and ending with a period
        if current_length >= self.num_tokens:
            self.condition(current_length == self.num_tokens)
            # Check if generated_tokens is not empty before accessing last element
            if self.generated_tokens and self.generated_tokens[-1].token_id in self.period_tokens:
                self.condition(True)
            else:
                self.condition(False) # Force failure if condition not met
            self.finish()
            return

        next_dist = self.context.next_token()

        # For the last token, force it to be one ending with period
        if current_length == self.num_tokens - 1:
            period_mask = self.period_tokens
            if not period_mask: # Handle case where period_tokens might be empty
                logger.error("period_tokens mask is empty for final token constraint.")
                self.condition(False) # Fail the particle if no valid tokens exist
                self.finish()
                return
            await self.observe(self.context.mask_dist(period_mask), True)

        # For non-final tokens, prevent period tokens and EOS
        else:
            # Create a full set of token IDs, then remove restricted ones
            all_token_ids = set(range(len(self.lm.vocab)))
            non_period_mask = all_token_ids - self.period_tokens - {self.lm.tokenizer.eos_token_id}
            if not non_period_mask: # Handle case where mask might be empty
                logger.error("non_period_mask is empty for non-final token constraint.")
                self.condition(False) # Fail the particle if no valid tokens exist
                self.finish()
                return
            await self.observe(self.context.mask_dist(non_period_mask), True)

        # Sample the next token
        token = await self.sample(next_dist)
        self.generated_tokens.append(token)


# The following are used for metrics around total server runtime
start_time = time.time()
total_requests = 0
total_tokens = 0
request_times = []

##### API ENDPOINTS AND HELPER METHODS #####

async def get_next_available_model():
    global lm_models, model_locks
    # With one model per worker, we always return the first (and only) model
    return 0, lm_models[0], model_locks[0]

async def generate_text(request):
    # Basic validation for num_particles and beam_factor
    if request.num_particles <= 0 or request.beam_factor <= 0:
        raise ValueError("num_particles and beam_factor must be positive integers.")
    
    # Check if lm_models is initialized
    if not lm_models:
        logger.error("Model not initialized. Cannot generate text.")
        raise RuntimeError("Model not initialized.")

    model_idx, lm, lock = await get_next_available_model()
    
    async with lock:
        with torch.inference_mode():
            model = FixedLengthSentenceModel(
                lm=lm,
                prompt=request.prompt,
                num_tokens=request.num_tokens,
                temperature=request.temperature
            )
            
            try:
                particles = await smc_steer(
                    model, 
                    request.num_particles, 
                    request.beam_factor
                )
                
                if not particles:
                    logger.warning(f"SMC steering returned no particles for request: {request.prompt}")
                    return {"generated_text": "Generation failed: No valid particles found."}

                best_particle = max(particles, key=lambda p: p.weight)
                # Ensure the context is converted to string
                generated_text = str(best_particle.context)
                return {"generated_text": generated_text}
            except Exception as e:
                logger.error(f"Error during SMC steering for worker {WORKER_ID}: {e}")
                raise

@app.get("/health")
async def health():
    if not lm_models:
        return {"status": "loading"}
    return {
        "status": "ready", 
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0 # This worker only sees GPU 0
    }

@app.get("/model_info")
async def model_info():
    if not lm_models:
        return {"status": "loading"}
    
    # Added checks for tokenizer attributes to prevent errors if not present
    eos_token_id = lm_models[0].tokenizer.eos_token_id if hasattr(lm_models[0].tokenizer, 'eos_token_id') else None
    max_model_len = lm_models[0].max_model_len if hasattr(lm_models[0], 'max_model_len') else None
    
    return {
        "eot_token_id": eos_token_id,
        "max_length": max_model_len, # Using max_model_len from the loaded model
        "worker_id": WORKER_ID,
        "gpu_id_in_container": 0 # This worker only sees GPU 0
    }

@app.get("/stats")
async def get_stats():
    global start_time, total_requests, total_tokens, request_times
    
    elapsed = time.time() - start_time
    # Ensure no division by zero for avg_latency
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
async def generate(request: GenerationRequest):
    global total_requests, total_tokens, request_times
    
    start_time_req = time.time()
    try:
        result = await generate_text(request)
        duration = time.time() - start_time_req
        
        request_times.append(duration)
        total_requests += 1
        # len(result["generated_text"]) might not be accurate for token count
        # For more accurate token count, you'd need to tokenize the output
        # For now, we'll keep it as is, but be aware of this for production metrics.
        total_tokens += len(result.get("generated_text", "")) 
        
        if total_requests % 10 == 0:
            elapsed = time.time() - start_time
            # Calculate rolling average latency for last 100 requests
            avg_latency = sum(request_times[-100:]) / min(100, len(request_times))
            logger.info(f"Worker {WORKER_ID} | GPU (local) {0} | Requests: {total_requests} | Avg latency (last {min(100, len(request_times))} req): {avg_latency:.4f}s")
        
        return result
    except Exception as e:
        logger.error(f"API Error in /generate for worker {WORKER_ID}: {e}", exc_info=True)
        # Return a meaningful error response to the client
        return {"error": str(e), "status_code": 500}