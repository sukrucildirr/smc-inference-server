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
from util.request_model import GenerationRequest


MODEL_NAME = "NousResearch/Hermes-3-Llama-3.2-3B"

########### parallel server config ##############

WORKER_ID = int(os.environ.get("WORKER_ID", 0))
NUM_GPUS = 1

#################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lm_models = []
current_model_idx = 0
model_locks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lm_models, model_locks

    gpu_id = WORKER_ID % NUM_GPUS
    logger.info(f"Worker {WORKER_ID} initializing model on GPU {gpu_id}")

    # Startup and initialization steps

    kwargs = (
        {"engine_opts": {"gpu_memory_utilization": 0.95, "max_model_len": 1024}}
    )
    
    logger.info(f"Loading model on GPU {gpu_id}...")
    lm = CachedCausalLM.from_pretrained(
        MODEL_NAME,
        backend='vllm',
        **kwargs
    )
    lm.batch_size = 1
    
    lm_models = [lm]
    model_locks = [asyncio.Lock()]

    # Model warmup
    
    logger.info(f"Warming up model on GPU {gpu_id}...")
    dummy_model = FixedLengthThinkingModel(lm=lm, prompt="Hello, world", num_tokens=20)
    await smc_steer(dummy_model, 3, 1)
    logger.info(f"Model warmup complete on GPU {gpu_id}")
    
    yield

    logger.info(f"Shutting down model on GPU {gpu_id}...")
    for lm in lm_models:
        del lm

app = FastAPI(lifespan=lifespan)


######## SMC CONSTRAINT MODEL ########
class FixedLengthThinkingModel(Model):
    """This FixedLengthThinkingModel demonstrates an example constraint to be used with the SMC Inference Server."""
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
        for i, token in enumerate(self.lm.vocab):
            if token.endswith('.'):
                self.period_tokens.add(i)

    async def step(self):
        """Generate exactly num_tokens tokens with a coherent sentence ending in period."""
        current_length = len(self.generated_tokens)
        
        if current_length >= self.num_tokens:
            # Condition on exact length and ending with a period
            self.condition(current_length == self.num_tokens)
            self.condition(self.generated_tokens[-1].token_id in self.period_tokens)
            self.finish()
            return

        next_dist = self.context.next_token()

        if current_length == self.num_tokens - 1:
            # For the last token, force it to be one ending with period
            period_mask = self.period_tokens
            await self.observe(self.context.mask_dist(period_mask), True)
        else:
            # For non-final tokens, prevent period tokens and EOS
            non_period_mask = set(range(len(self.lm.vocab))) - self.period_tokens - {self.lm.tokenizer.eos_token_id}
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
    global lm_models, model_locks, current_model_idx
    return 0, lm_models[0], model_locks[0]

async def generate_text(request):
    model_idx, lm, lock = await get_next_available_model()
    
    async with lock:
        with torch.inference_mode():
            model = FixedLengthThinkingModel(
                lm=lm,
                prompt=request.prompt,
                num_tokens=request.num_tokens,
                temperature=request.temperature
            )
            
            particles = await smc_steer(
                model, 
                request.num_particles, 
                request.beam_factor
            )
            
            best_particle = max(particles, key=lambda p: p.weight)
            return {"generated_text": str(best_particle.context)}

@app.get("/health")
async def health():
    if not lm_models:
        return {"status": "loading"}
    return {
        "status": "ready", 
        "worker_id": WORKER_ID,
        "gpu_id": WORKER_ID % NUM_GPUS
    }

@app.get("/model_info")
async def model_info():
    if not lm_models:
        return {"status": "loading"}
    
    return {
        "eot_token_id": lm_models[0].tokenizer.eos_token_id,
        "max_length": 8192,
        "worker_id": WORKER_ID,
        "gpu_id": WORKER_ID % NUM_GPUS
    }

@app.get("/stats")
async def get_stats():
    global start_time, total_requests, total_tokens, request_times
    
    elapsed = time.time() - start_time
    avg_latency = sum(request_times) / len(request_times) if request_times else 0
    
    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
        "avg_latency": avg_latency,
        "uptime_seconds": elapsed,
        "worker_id": WORKER_ID,
        "gpu_id": WORKER_ID % NUM_GPUS
    }

@app.post("/generate")
async def generate(request: GenerationRequest):
    global total_requests, total_tokens, request_times
    
    start_time_req = time.time()
    result = await generate_text(request)
    duration = time.time() - start_time_req
    
    request_times.append(duration)
    total_requests += 1
    total_tokens += len(result["generated_text"])
    
    if total_requests % 10 == 0:
        elapsed = time.time() - start_time
        avg_latency = sum(request_times[-100:]) / min(100, len(request_times))
        logger.info(f"Worker {WORKER_ID} | GPU {WORKER_ID % NUM_GPUS} | Requests: {total_requests} | Avg latency: {avg_latency:.4f}s")
    
    return result