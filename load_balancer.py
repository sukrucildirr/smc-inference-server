from fastapi import FastAPI, Request, BackgroundTasks
import httpx
import asyncio
from typing import Dict, Any
import time
import torch

app = FastAPI()

BASE_PORT = 8000

# Get the number of GPUs
num_gpus = torch.cuda.device_count()

# Generate the list of backend servers
BACKEND_SERVERS = [f"http://localhost:{BASE_PORT + i + 1}" for i in range(num_gpus)]

print(BACKEND_SERVERS)

current_index = 0
lock = asyncio.Lock()

server_status: Dict[str, Dict[str, Any]] = {
    server: {"busy": False, "last_used": 0} for server in BACKEND_SERVERS
}

async def get_next_available_server():

    global current_index
    
    async with lock:
        for _ in range(len(BACKEND_SERVERS)):
            server = BACKEND_SERVERS[current_index]
            current_index = (current_index + 1) % len(BACKEND_SERVERS)
            
            if not server_status[server]["busy"]:
                server_status[server]["busy"] = True
                server_status[server]["last_used"] = time.time()
                return server
        
        least_recent = min(BACKEND_SERVERS, key=lambda s: server_status[s]["last_used"])
        server_status[least_recent]["last_used"] = time.time()
        return least_recent

async def forward_request(request: Request, backend_url: str, endpoint: str):
    try:
        async with httpx.AsyncClient() as client:
            url = f"{backend_url}{endpoint}"
            if request.method == "POST":
                req_data = await request.json()
                response = await client.post(url, json=req_data, timeout=300.0)
            else:
                response = await client.get(url, timeout=5.0)
            
            return response.json()
    except Exception as e:
        return {"error": f"Backend {backend_url} failed: {str(e)}"}
    finally:
        if backend_url in server_status:
            server_status[backend_url]["busy"] = False

@app.post("/generate")
async def generate(request: Request, background_tasks: BackgroundTasks):
    """Load balancing with true parallelism."""
    backend = await get_next_available_server()
    response = await forward_request(request, backend, "/generate")
    return response

@app.get("/health")
async def health_check():
    async def check_backend(backend):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{backend}/health", timeout=2.0)
                return {"server": backend, "status": "up", "details": response.json()}
        except Exception as e:
            return {"server": backend, "status": "down", "error": str(e)}
    
    tasks = [check_backend(backend) for backend in BACKEND_SERVERS]
    results = await asyncio.gather(*tasks)
    
    return {
        "load_balancer": "healthy",
        "backends": results
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application state."""
    print(f"Load balancer started with {len(BACKEND_SERVERS)} backend servers")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        workers=4,
        limit_concurrency=32,
    )