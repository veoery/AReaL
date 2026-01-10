# AlfWorld Task Training with AReaL

This directory contains the implementation for training language models on the AlfWorld household task using reinforcement learning.

## Overview

AlfWorld is a text-based interactive environment where an agent navigates household environments and manipulates objects to complete tasks like:
- Put objects in locations (pick_and_place)
- Clean then place objects (pick_clean_then_place)
- Heat then place objects (pick_heat_then_place)
- Cool then place objects (pick_cool_then_place)
- Examine objects (look_at_obj)
- Pick two objects (pick_two_obj)

## Setup

### 1. Docker Setup

AlfWorld runs in Docker (automatically handled by task_server_adapter). Make sure you have:
- Docker installed and running
- The AlfWorld Docker image: `longinyu/agentbench-alfworld`

Pull the image if needed:
```bash
docker pull longinyu/agentbench-alfworld
```

The task_server_adapter will automatically launch the Docker container when you start the server.

### 2. Generate Training Indices

**Note**: The index files (`train_indices.jsonl` and `valid_indices.jsonl`) have been pre-generated with standard dataset sizes:
- Training: 134 samples (standard split)
- Validation: 24 samples (dev split)

If your AlfWorld dataset has different sizes, you'll need to regenerate indices:

```bash
cd /home/haizhonz/yizhuod/AReaL
python examples/alfworld/generate_indices_simple.py
```

Adjust `NUM_STANDARD_GAMES` and `NUM_DEV_GAMES` in the script if needed.

## Running Training

### Step 1: Start Task Servers

Start two task servers in separate terminals:

```bash
# Terminal 1: Training server (standard split)
# This will automatically launch Docker container with alfworld-std
cd /home/haizhonz/yizhuod/AgentBench
python -m src.server.task_server_adapter alfworld-std --port 5000

# Terminal 2: Evaluation server (dev split)
# This will automatically launch Docker container with alfworld-dev
cd /home/haizhonz/yizhuod/AgentBench
python -m src.server.task_server_adapter alfworld-dev --port 5001
```

**Note**: The task_server_adapter detects that AlfWorld requires Docker (from the config) and automatically launches a Docker container. You don't need to manually run docker commands.

### Step 2: Start Training

```bash
# Terminal 3: Training process
cd /home/haizhonz/yizhuod/AReaL
conda activate <your-areal-environment>

python -m areal.launcher.local examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    experiment_name=alfworld_rl \
    trial_name=run1
```

## Configuration

### Key Parameters

Edit `config.yaml` to adjust training:

- **GPU Allocation**: `allocation_mode: sglang:d8p1t1+d2p1t1` (8 GPUs for rollout, 2 for training)
- **Max Turns**: Set in `train.py` as `max_turns=35` (AlfWorld uses 35 max steps)
- **Failure Penalty**: `failure_penalty=-0.1` in `train.py` (negative reward for failures)
- **Turn Discount**: `turn_discount=0.95` (penalize taking more turns)

### WandB Logging

Set environment variables for WandB:

```bash
export WANDB_API_KEY="your-api-key"
export WANDB_BASE_URL="https://your-wandb-instance.com"  # Optional

# Or configure via command line
python -m areal.launcher.local examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    experiment_name=alfworld_rl \
    trial_name=run1 \
    stats_logger.wandb.project=my-project \
    stats_logger.wandb.entity=my-username
```

## Implementation Details

### Custom Workflow

The `AlfWorldWorkflow` class in `train.py` implements:

1. **Negative Rewards**: Failures get `-0.1` reward instead of `0.0`
2. **Turn Discount**: Reward decays as `reward * (0.95 ^ num_turns)`
3. **Error Handling**: Invalid actions trigger feedback messages instead of termination

### Modified Task Code

The AlfWorld task in `AgentBench/src/server/tasks/alfworld/task.py` has been modified to:
- Give error feedback when action parsing fails
- Continue episode instead of terminating on invalid format
- Inject format instructions to help agent learn

## Files

```
examples/alfworld/
├── README.md                           # This file
├── train.py                            # Training script with AlfWorldWorkflow
├── config.yaml                         # Training configuration
├── generate_indices_simple.py          # Script to generate index files
└── data/
    ├── train_indices.jsonl             # Training sample indices (std-00000 to std-00133)
    └── valid_indices.jsonl             # Validation sample indices (dev-00000 to dev-00023)
```

## Troubleshooting

### Docker image not found

Pull the AlfWorld Docker image:
```bash
docker pull longinyu/agentbench-alfworld
```

### "Sample std-XXXXX not found in task alfworld-std"

The number of samples in your dataset doesn't match the indices. Regenerate indices using `generate_indices_simple.py` with correct counts.

### Task server returns 404 errors

Make sure:
1. Task servers are running on correct ports (5000 for training, 5001 for eval)
2. Index prefixes match task names (`std-` for alfworld-std, `dev-` for alfworld-dev)
3. Sample counts don't exceed actual dataset size

## Comparison with OS Interaction

| Aspect | AlfWorld | OS Interaction |
|--------|----------|----------------|
| Environment | Text-based household simulation | Docker bash shell |
| Actions | Natural language (e.g., "take apple") | Shell commands |
| Max Turns | 35 | 8 |
| Docker | Yes (longinyu/agentbench-alfworld) | Yes (task creates containers) |
| Data Size | ~134 training, ~24 eval | ~144 training, ~26 eval |
| Action Format | THOUGHT + ACTION | Think + Act: bash |
| Task Server | Runs in Docker container | Runs directly, creates Docker containers for tasks |
