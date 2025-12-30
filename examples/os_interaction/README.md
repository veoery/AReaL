# OS Interaction Task Training

This example demonstrates how to train language models on the OS Interaction task using AReaL's Task Server Architecture.

## Overview

The Task Server Architecture separates environment logic from RL training:

- **Task Server** (AgentBench): Manages OS environments (Docker containers), executes bash commands, evaluates answers
- **AReaL Training**: Handles model inference, trajectory collection, PPO updates

This separation allows:
- Task developers to focus on environment logic without understanding RL
- AReaL users to train on any task implementing the standard API
- Independent development and deployment of tasks

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│  AReaL Training     │  HTTP   │  Task Server        │
│  - Model inference  │◄───────►│  - Docker containers│
│  - PPO training     │         │  - Bash execution   │
│  - Trajectory       │         │  - Reward evaluation│
└─────────────────────┘         └─────────────────────┘
```

## Prerequisites

1. **AgentBench Setup**
   ```bash
   cd /path/to/AgentBench
   conda create -n agentbench python=3.9
   conda activate agentbench
   pip install -r requirements.txt
   ```

2. **Docker Images**
   ```bash
   # Build OS interaction Docker images
   docker build -f data/os_interaction/res/dockerfiles/default \
       data/os_interaction/res/dockerfiles \
       --tag local-os/default
   ```

3. **AReaL Setup**
   ```bash
   cd /path/to/AReaL
   # Install AReaL dependencies (see main README)
   ```

## Quick Start

### Step 1: Start Task Server

In one terminal:

```bash
cd /path/to/AgentBench
conda activate agentbench

# Start OS task server on port 5000
python -m src.server.task_server_adapter os-dev --port 5000
```

You should see:
```
======================================================================
Starting Task Server: os-dev
======================================================================
Server URL: http://0.0.0.0:5000
API Base:   http://0.0.0.0:5000/api
Health:     http://0.0.0.0:5000/api/health
Task Info:  http://0.0.0.0:5000/api/task/info
======================================================================
```

### Step 2: Verify Task Server

Test the server is working:

```bash
# Get task info
curl http://localhost:5000/api/task/info

# Start an episode
curl -X POST http://localhost:5000/api/episode/start \
  -H "Content-Type: application/json" \
  -d '{"sample_id": "dev-001-00000", "config": {}}'
```

### Step 3: Start Training

In another terminal:

```bash
cd /path/to/AReaL

# Single node training
python -m areal.launcher.local examples/os_interaction/train.py \
    --config examples/os_interaction/config.yaml \
    experiment_name=os_rl_experiment \
    trial_name=run1
```

## Configuration

Edit `config.yaml` to customize training:

### Task Server Connection

```yaml
task_server_url: "http://localhost:5000"  # Task server address
```

### Model Configuration

```yaml
model_path: "meta-llama/Llama-3.2-1B-Instruct"
tokenizer_path: "meta-llama/Llama-3.2-1B-Instruct"
```

### Generation Settings

```yaml
gconfig:
  temperature: 1.0      # Higher = more exploration
  max_tokens: 512       # Max tokens per turn
  n_samples: 4          # Trajectories per sample (diversity)
```

### PPO Hyperparameters

```yaml
ppo:
  learning_rate: 1e-5
  clip_range: 0.2
  num_epochs: 3
```

### Training Settings

```yaml
training:
  num_iterations: 1000
  rollout_batch_size: 32  # Episodes per iteration
  eval_interval: 5
```

## Dataset Format

The dataset just needs to provide sample IDs that the task server understands:

```jsonl
{"index": "dev-001-00000", "messages": []}
{"index": "dev-001-00001", "messages": []}
```

The `messages` field is empty because the task server provides the initial prompt.

## Task Server API

The task server implements a standard API:

### Start Episode
```bash
POST /api/episode/start
{
  "sample_id": "dev-001-00000",
  "config": {}
}

Response:
{
  "episode_id": "uuid-1234",
  "observation": {
    "type": "text",
    "content": "You are an OS assistant. Task: count files in /etc"
  },
  "info": {"max_turns": 8, "sample_id": "dev-001-00000"}
}
```

### Step Episode
```bash
POST /api/episode/step
{
  "episode_id": "uuid-1234",
  "action": {
    "type": "text",
    "content": "Think: Use ls. Act: bash\n```bash\nls /etc | wc -l\n```"
  }
}

Response:
{
  "episode_id": "uuid-1234",
  "observation": {
    "type": "text",
    "content": "The output of the OS:\n220"
  },
  "reward": 0.0,
  "done": false,
  "info": {"turn": 1}
}
```

See [docs/task_server_api.md](../../docs/task_server_api.md) for full specification.

## Customizing the Workflow

If you need custom observation/action formatting, subclass `TaskServerWorkflow`:

```python
from areal.workflow.task_server import TaskServerWorkflow

class CustomOSWorkflow(TaskServerWorkflow):
    def format_observation_to_messages(self, observation, history):
        # Custom formatting
        content = observation["content"]
        # Add special instructions, etc.
        return history + [{"role": "user", "content": content}]

    def postprocess_reward(self, reward, num_turns, info):
        # Custom reward shaping
        if info.get("success"):
            # Bonus for efficiency
            return reward + (1.0 / num_turns)
        return reward
```

Then use it in your training script:

```python
workflow = CustomOSWorkflow(
    task_server_url="http://localhost:5000",
    ...
)
```

## Monitoring

### Task Server Logs

The task server shows:
- Episode starts
- Action parsing
- Bash execution
- Rewards
- Errors

### AReaL Logs

Training metrics are logged to:
- Console (real-time)
- Weights & Biases (if configured)
- TensorBoard (if configured)

Key metrics:
- `reward`: Episode rewards
- `num_turns`: Turns per episode
- `success`: Success rate

### Health Check

```bash
curl http://localhost:5000/api/health
```

Returns:
```json
{
  "status": "healthy",
  "active_episodes": 0,
  "task_name": "os-dev"
}
```

## Troubleshooting

### Task Server Not Reachable

```
Error: Failed to connect to task server http://localhost:5000
```

**Solution**: Make sure task server is running:
```bash
python -m src.server.task_server_adapter os-dev --port 5000
```

### Docker Permission Denied

```
Error: Got permission denied while trying to connect to Docker daemon
```

**Solution**: Add user to docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Episode Timeout

```
Error: Request to http://localhost:5000/api/episode/step timed out
```

**Possible causes**:
- Docker container is slow
- Bash command hangs
- Server overloaded

**Solution**: Increase timeout in config:
```python
workflow = TaskServerWorkflow(
    task_server_url="http://localhost:5000",
    timeout=120.0,  # Increase to 120 seconds
    ...
)
```

## Advanced Usage

### Multiple Task Workers

For better throughput, run multiple task workers:

```bash
# Terminal 1: Worker on port 5000
python -m src.server.task_server_adapter os-dev --port 5000

# Terminal 2: Worker on port 5001
python -m src.server.task_server_adapter os-dev --port 5001

# Terminal 3: Worker on port 5002
python -m src.server.task_server_adapter os-dev --port 5002
```

Then use a load balancer (nginx, haproxy) to distribute requests.

### Different Task Types

To train on other AgentBench tasks:

```bash
# Database task
python -m src.server.task_server_adapter dbbench-dev --port 5000

# Knowledge graph task
python -m src.server.task_server_adapter kg-dev --port 5000
```

Update the dataset indices to match the task samples.

### Custom Tasks

To add your own task:

1. Implement the Task Server API (any language/framework)
2. Return observations in `{"type": "text", "content": "..."}` format
3. Compute rewards and return in responses
4. Point AReaL to your server URL

No changes to AReaL code needed!

## References

- [Task Server API Specification](../../docs/task_server_api.md)
- [AReaL Documentation](../../README.md)
- [AgentBench Paper](https://arxiv.org/abs/2308.03688)
