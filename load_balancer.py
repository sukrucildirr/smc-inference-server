from fastapi import FastAPI, Request, BackgroundTasks
import httpx
import asyncio
from typing import Dict, Any
import time
import os

# Initialize FastAPI app
app = FastAPI()

# --- Configuration ---
# Number of backend workers, configurable via environment variable, defaults to 2
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
# Base port for backend servers, configurable via environment variable, defaults to 8001
BACKEND_BASE_PORT = int(os.environ.get("BACKEND_BASE_PORT", 8001))

# Generate a list of backend server URLs
BACKEND_SERVERS = [
    f"http://localhost:{BACKEND_BASE_PORT + i}" for i in range(NUM_WORKERS)
]

# --- Load Balancing State ---
# Global index to keep track of the current backend server for round-robin
current_index = 0
# Asynchronous lock to protect access to shared state (current_index and server_status)
lock = asyncio.Lock()

# Dictionary to store the status of each backend server, including busy state and last usage time
server_status: Dict[str, Dict[str, Any]] = {
    server: {"busy": False, "last_used": 0} for server in BACKEND_SERVERS
}

# --- Helper Functions ---

async def get_next_available_server() -> str:
    """
    Selects the next available backend server using a round-robin approach.
    If all servers are busy, it returns the server that was least recently used.

    Returns:
        str: The URL of the selected backend server.
    """
    global current_index

    async with lock:
        # Iterate through servers to find a non-busy one
        for _ in range(len(BACKEND_SERVERS)):
            server = BACKEND_SERVERS[current_index]
            current_index = (current_index + 1) % len(BACKEND_SERVERS)

            if not server_status[server]["busy"]:
                server_status[server]["busy"] = True
                server_status[server]["last_used"] = time.time()
                return server

        # If all servers are busy, find the least recently used server
        least_recent = min(BACKEND_SERVERS, key=lambda s: server_status[s]["last_used"])
        server_status[least_recent]["busy"] = True  # Mark as busy even if it was already
        server_status[least_recent]["last_used"] = time.time()
        return least_recent

async def forward_request(request: Request, backend_url: str, endpoint: str) -> Dict[str, Any]:
    """
    Forwards the incoming request to the specified backend server.

    Args:
        request (Request): The incoming FastAPI request object.
        backend_url (str): The base URL of the backend server.
        endpoint (str): The specific API endpoint on the backend to call (e.g., "/generate").

    Returns:
        Dict[str, Any]: The JSON response from the backend, or an error dictionary.
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"{backend_url}{endpoint}"
            if request.method == "POST":
                req_data = await request.json()
                response = await client.post(url, json=req_data, timeout=300.0)  # Long timeout for generation
            else:
                # Default timeout for GET requests
                response = await client.get(url, timeout=5.0)

            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
    except httpx.RequestError as e:
        return {"error": f"Backend {backend_url} request failed: {type(e).__name__} - {e}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Backend {backend_url} responded with an error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred while forwarding to {backend_url}: {str(e)}"}
    finally:
        # Ensure the server is marked as not busy after the request is processed
        if backend_url in server_status:
            server_status[backend_url]["busy"] = False

# --- API Endpoints ---

@app.post("/generate")
async def generate(request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Handles requests for content generation by forwarding them to an available backend server.
    This endpoint demonstrates load balancing with "true parallelism" as requests can be
    processed concurrently by different backend servers.

    Args:
        request (Request): The incoming request containing data for generation.
        background_tasks (BackgroundTasks): FastAPI's dependency for adding background tasks.
                                            Used implicitly by the forward_request's finally block
                                            to release the server.

    Returns:
        Dict[str, Any]: The response from the backend server.
    """
    backend = await get_next_available_server()
    response = await forward_request(request, backend, "/generate")
    return response

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Provides a health check for the load balancer and all registered backend servers.
    It concurrently checks the health of each backend.

    Returns:
        Dict[str, Any]: A dictionary containing the health status of the load balancer
                        and each backend server.
    """
    async def check_backend(backend: str) -> Dict[str, Any]:
        """
        Internal helper to check the health of a single backend server.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{backend}/health", timeout=2.0)
                response.raise_for_status()
                return {"server": backend, "status": "up", "details": response.json()}
        except httpx.RequestError as e:
            return {"server": backend, "status": "down", "error": f"Connection error: {type(e).__name__} - {e}"}
        except httpx.HTTPStatusError as e:
            return {"server": backend, "status": "down", "error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"server": backend, "status": "down", "error": f"An unexpected error: {str(e)}"}

    # Run health checks for all backends concurrently
    tasks = [check_backend(backend) for backend in BACKEND_SERVERS]
    results = await asyncio.gather(*tasks)

    return {
        "load_balancer": "healthy",
        "backends": results
    }

# --- Application Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """
    Initializes application state and logs information when the FastAPI application starts up.
    """
    print(f"Load balancer started with {len(BACKEND_SERVERS)} backend servers")
    print(f"Backend servers configured: {BACKEND_SERVERS}")


# --- Main Execution ---

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using Uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all available network interfaces
        port=8000,       # Load balancer will listen on port 8000
        workers=4,       # Uvicorn workers for handling incoming requests to the load balancer
        limit_concurrency=32, # Limit the number of concurrent connections (requests) per worker
    )