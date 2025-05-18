#!/bin/bash

# Cleanup existing containers
docker ps -a --filter "name=llamppl-worker" --format "{{.Names}}" | while read -r name; do
    echo "Stopping and removing container: $name"
    docker stop "$name" > /dev/null 2>&1
    docker rm "$name" > /dev/null 2>&1
done

# Define the number of workers and base host port
NUM_WORKERS=2
BASE_HOST_PORT=8001 # Start mapping from host port 8001

# Delay in seconds between launching each worker
LAUNCH_DELAY_SECONDS=120 # <--- Start with a generous delay like 60 seconds

echo "Starting ${NUM_WORKERS} llamppl inference workers..."

for i in $(seq 0 $((NUM_WORKERS-1))); do
    HOST_PORT=$((BASE_HOST_PORT + i))
    DEVICE_ID=$i # Assumes your GPUs are indexed 0, 1, 2, ...
    CONTAINER_NAME="llamppl-worker-${i}"
    RANK=$i

    echo "Launching worker ${i} on physical GPU ${DEVICE_ID} (mapped to container's /dev/cuda:0)..."
    docker run -d \
      --name "${CONTAINER_NAME}" \
      --gpus "device=${DEVICE_ID}" \
      -p "${HOST_PORT}":8000 \
      -e WORKER_ID="${i}" \
      -e CUDA_VISIBLE_DEVICES="0" \
      -e VLLM_NO_CUDA_GRAPHS="1" \
      --shm-size="4g" \
      -v "$(pwd)/.model_cache:/app/huggingface" \
      llamppl-inference-server

    if [ "$i" -lt $((NUM_WORKERS-1)) ]; then
        echo "Waiting for ${LAUNCH_DELAY_SECONDS} seconds before launching next worker..."
        sleep ${LAUNCH_DELAY_SECONDS}
    fi
done

echo "All workers launched. Use 'docker ps' to check status."
echo "You can check individual worker logs with 'docker logs llamppl-worker-X'"