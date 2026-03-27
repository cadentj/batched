
## Setup & Lifecycle
```bash
# Set these before starting the daemon AND before launching clients
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Start daemon
nvidia-cuda-mps-control -d

# Check status
ps -ef | grep nvidia-cuda-mps

# Stop daemon
echo quit | nvidia-cuda-mps-control
```

## Launching a Python Client

Any CUDA process automatically becomes an MPS client if the daemon is running.
The client just needs to see the same env vars:
```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
python my_inference.py
```

## Soft Partitioning (Volta+)

Caps SM usage per client. Partitions can overlap.
```bash
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 \
python my_inference.py
```

## Static Partitioning (Ampere+)

Assigns disjoint SM chunks per client. Must start daemon with `-S`:
```bash
# Start daemon in static mode
nvidia-cuda-mps-control -d -S

# Create a partition (7 chunks)
echo "sm_partition add <GPU-UUID> 7" | nvidia-cuda-mps-control
# → returns <partition-ID>

# Launch client on that partition
CUDA_VISIBLE_DEVICES=0 \
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
CUDA_MPS_SM_PARTITION=<GPU-UUID>/<partition-ID> \
python my_inference.py

# List partitions
echo "lspart" | nvidia-cuda-mps-control

# Remove partition (kill clients first)
echo "sm_partition rm <GPU-UUID>/<partition-ID>" | nvidia-cuda-mps-control
```

## Reset
```bash
echo quit | nvidia-cuda-mps-control
# then restart with or without -S
nvidia-cuda-mps-control -d
```