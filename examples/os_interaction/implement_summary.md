# Task Server Architecture Implementation Summary

## Overview

This implementation provides a **clean separation** between task environments and RL training, enabling:

1. **Task developers** to focus on environment logic without understanding RL
2. **AReaL users** to train on any task via a standardized HTTP API
3. **Independent deployment** of tasks and training infrastructure

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      AReaL Training Side                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  examples/os_interaction/train.py                                  │
│    ↓                                                               │
│  areal/workflow/task_server.py (TaskServerWorkflow)               │
│    - HTTP client to task server                                   │
│    - Trajectory collection                                        │
│    - Tensor construction                                          │
│    ↓                                                               │
│  PPOTrainer                                                        │
│    - Model inference (SGLang/vLLM)                                │
│    - PPO updates                                                   │
│                                                                     │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
                          │ HTTP (Standard API)
                          │
┌─────────────────────────▼──────────────────────────────────────────┐
│                      Task Server Side                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AgentBench/src/server/task_server_adapter.py                     │
│    - Wraps AgentBench tasks                                       │
│    - Implements standard API                                      │
│    - Episode lifecycle management                                 │
│    ↓                                                               │
│  AgentBench/src/server/tasks/os_interaction/task.py              │
│    - Docker container management                                  │
│    - Bash command execution                                       │
│    - Reward evaluation                                            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Standard Task Server API (`docs/task_server_api.md`)

Defines the HTTP API that all task servers must implement:

**Endpoints:**
- `GET /api/task/info` - Task metadata
- `POST /api/episode/start` - Initialize episode
- `POST /api/episode/step` - Execute action
- `POST /api/episode/cancel` - Cancel episode
- `GET /api/health` - Health check

**Key Design Principles:**
- Language-agnostic (HTTP-based)
- Stateful episodes (Docker containers, game state, etc.)
- Observation/action abstraction
- Timeout and cleanup handling

### 2. AgentBench Adapter (`AgentBench/src/server/task_server_adapter.py`)

Wraps existing AgentBench tasks with the standard API.

**Features:**
- Converts AgentBench's Session mechanism to HTTP API
- Episode state tracking (active episodes, timeouts)
- Automatic cleanup of expired episodes
- Error handling and cancellation

**Usage:**
```bash
python -m src.server.task_server_adapter os-dev --port 5000
```

**Key Classes:**
- `TaskServerAdapter`: Main adapter logic
- `EpisodeState`: Tracks running episodes
- `FastAPI app`: HTTP server

### 3. Generic TaskServerWorkflow (`areal/workflow/task_server.py`)

AReaL's generic workflow for connecting to any task server.

**Features:**
- HTTP client with timeout handling
- Multi-turn trajectory collection
- Token-level tracking for RL
- Reward shaping (turn discount)
- Customization hooks

**Customization Points:**
```python
class CustomWorkflow(TaskServerWorkflow):
    def format_observation_to_messages(self, observation, history):
        # Custom prompt formatting
        pass

    def format_agent_output_to_action(self, agent_output, info):
        # Custom action formatting
        pass

    def postprocess_reward(self, reward, num_turns, info):
        # Custom reward shaping
        pass
```

### 4. Training Script (`examples/os_interaction/train.py`)

Simple training script that connects everything.

**Key Points:**
- Just 40 lines of code!
- Configuration-driven
- Works with any task server
- No task-specific code needed

## Usage Flow

### Setup (One-Time)

1. **AgentBench Setup:**
   ```bash
   cd AgentBench
   conda create -n agentbench python=3.9
   conda activate agentbench
   pip install -r requirements.txt

   # Build Docker images
   docker build -f data/os_interaction/res/dockerfiles/default \
       data/os_interaction/res/dockerfiles \
       --tag local-os/default
   ```

2. **AReaL Setup:**
   ```bash
   cd AReaL
   # Install AReaL (see main README)
   ```

### Training Session

**Terminal 1: Start Task Server**
```bash
cd AgentBench
conda activate agentbench
python -m src.server.task_server_adapter os-dev --port 5000
```

**Terminal 2: Test Connection (Optional)**
```bash
cd AReaL
python examples/os_interaction/test_connection.py --server http://localhost:5000
```

**Terminal 3: Start Training**
```bash
cd AReaL
python -m areal.launcher.local examples/os_interaction/train.py \
    --config examples/os_interaction/config.yaml \
    experiment_name=os_rl \
    trial_name=run1
```

## Benefits

### For Task Developers

✓ **Focus on domain logic**
  - No need to understand RL algorithms
  - No PyTorch dependencies
  - No distributed training complexity

✓ **Use any tech stack**
  - Python, JavaScript, Rust, Java, etc.
  - Just implement HTTP API

✓ **Independent testing**
  - Test with curl/Postman
  - Debug without RL framework
  - Deploy separately

### For AReaL Users

✓ **Train on any task**
  - Just point to task server URL
  - No code changes to AReaL

✓ **Easy experimentation**
  - Swap tasks by changing URL
  - Try different environments

✓ **Scalability**
  - Run multiple task workers
  - Load balance across servers

### For the Ecosystem

✓ **Community contributions**
  - Publish task servers as Docker images
  - Share via Docker Hub / GitHub
  - No need to modify AReaL core

✓ **Language diversity**
  - Task servers in any language
  - Best tool for each task

✓ **Modularity**
  - Update tasks independently
  - Update AReaL independently

## Implementation Details

### Trajectory Construction

The workflow collects token-level data across multiple turns:

```python
# Turn 1: Prompt + Completion
[prompt_tokens_1] [completion_tokens_1]

# Turn 2: New prompt tokens + Completion
# (History is appended, so we only take NEW prompt tokens)
[new_prompt_tokens_2] [completion_tokens_2]

# Turn 3: ...
[new_prompt_tokens_3] [completion_tokens_3]

# Final sequence for training:
[prompt_1] [completion_1] [new_prompt_2] [completion_2] [new_prompt_3] [completion_3]
   ↑           ↑              ↑               ↑              ↑               ↑
 mask=0     mask=1         mask=0         mask=1         mask=0         mask=1
```

**Key insight:** We avoid duplicating history by tracking sequence length and only taking new tokens.

### Reward Handling

```python
# Server returns reward per step
step_1: reward = 0.0  (intermediate)
step_2: reward = 0.0  (intermediate)
step_3: reward = 1.0  (terminal, correct answer)

# Total reward
total_reward = 0.0 + 0.0 + 1.0 = 1.0

# Apply turn discount (penalize taking more turns)
shaped_reward = 1.0 * (0.95 ^ 3) = 0.857

# Assigned to entire trajectory for PPO
```

### Error Handling

- **Episode timeout**: Cleaned up after 300s inactivity
- **Server unreachable**: Workflow returns None (trajectory rejected)
- **Episode cancellation**: Automatic cleanup on errors
- **Docker issues**: Task server handles, doesn't crash training

## Files Created

```
AReaL/
├── docs/
│   └── task_server_api.md                    # API specification
├── areal/
│   └── workflow/
│       └── task_server.py                     # Generic workflow
└── examples/
    └── os_interaction/
        ├── README.md                          # Usage guide
        ├── IMPLEMENTATION_SUMMARY.md          # This file
        ├── train.py                           # Training script
        ├── config.yaml                        # Configuration
        ├── test_connection.py                 # Test script
        └── data/
            ├── train_indices.jsonl            # Training samples
            └── valid_indices.jsonl            # Validation samples

AgentBench/
└── src/
    └── server/
        └── task_server_adapter.py             # AgentBench adapter
```

## Extensibility

### Adding New Tasks

**Option 1: Use AgentBench Tasks**
```bash
# Start different task
python -m src.server.task_server_adapter dbbench-dev --port 5000
python -m src.server.task_server_adapter kg-dev --port 5001
```

**Option 2: Implement Custom Task Server**
```python
# my_task_server.py (any language/framework)
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/episode/start")
async def start_episode(request):
    # Your task initialization
    return {"episode_id": ..., "observation": ...}

@app.post("/api/episode/step")
async def step_episode(request):
    # Your action execution
    return {"observation": ..., "reward": ..., "done": ...}
```

### Customizing Workflows

For task-specific formatting:

```python
# areal/workflow/os_interaction.py
from areal.workflow.task_server import TaskServerWorkflow

class OSInteractionWorkflow(TaskServerWorkflow):
    def format_observation_to_messages(self, observation, history):
        # Add OS-specific instructions
        content = "OS Interaction Mode\n\n" + observation["content"]
        return history + [{"role": "user", "content": content}]

    def postprocess_reward(self, reward, num_turns, info):
        # Efficiency bonus
        if info.get("success") and num_turns <= 3:
            return reward + 0.5
        return reward
```

## Performance Considerations

### HTTP Overhead

- ~1-2ms per request (local)
- Negligible compared to:
  - Model inference: 100-1000ms
  - Docker command execution: 10-100ms

### Scalability

- Single task worker: ~10-50 concurrent episodes
- Multiple workers: Linear scaling
- Load balancer: Distribute across workers

### Optimization Tips

1. **Keep task server and AReaL on same machine** for local development
2. **Use multiple workers** for production training
3. **Monitor episode timeouts** - adjust if needed
4. **Profile task execution** - optimize slow operations

## Testing

### Manual Testing

```bash
# 1. Start server
python -m src.server.task_server_adapter os-dev --port 5000

# 2. Run test script
python examples/os_interaction/test_connection.py

# 3. Manual curl tests
curl http://localhost:5000/api/task/info
```

### Integration Testing

```bash
# Test with small dataset
python -m areal.launcher.local examples/os_interaction/train.py \
    --config examples/os_interaction/config.yaml \
    training.num_iterations=5 \
    training.rollout_batch_size=4
```

## Future Enhancements

### Short Term

- [ ] Support for other observation types (images, JSON)
- [ ] Batch episode starting (reduce HTTP round trips)
- [ ] Streaming observations (for long-running commands)
- [ ] Task server health monitoring and auto-restart

### Long Term

- [ ] Task server discovery service
- [ ] Task marketplace (Docker images)
- [ ] Multi-task curriculum learning
- [ ] Distributed task execution (tasks on different machines)
- [ ] WebSocket support for real-time feedback

## Troubleshooting

See [README.md](README.md#troubleshooting) for common issues and solutions.

## References

- [Task Server API Specification](../../docs/task_server_api.md)
- [AgentBench Paper](https://arxiv.org/abs/2308.03688)
- [AReaL Documentation](../../README.md)

---

**Questions or Issues?**

- Task Server API: See `docs/task_server_api.md`
- AgentBench Adapter: See `AgentBench/src/server/task_server_adapter.py`
- Workflow Implementation: See `areal/workflow/task_server.py`
- Usage Guide: See `examples/os_interaction/README.md`
