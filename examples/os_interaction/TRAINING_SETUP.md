# OS Interaction Training Setup

## Dataset Configuration

The training now uses datasets 1-6 for training and dataset 7 for evaluation.

### Training Data (datasets 1-6):
- **Dataset 1**: stock.json (1 problem)
- **Dataset 2**: environment.json (1 problem)
- **Dataset 3**: ac.json (1 problem)
- **Dataset 4**: 10 problems (N11, N225, N37, N4, N41, Q09, Q19, Q30, Q47, Q49)
- **Dataset 5**: new.json (1 problem)
- **Dataset 6**: new.json (1 problem)
- **Total**: 15 training problems

### Evaluation Data (dataset 7):
- **Dataset 7**: bootstrap.json (1 problem)
- **Total**: 1 evaluation problem

## File Structure

```
AReaL/examples/os_interaction/
├── data/
│   ├── train_indices.jsonl  # 15 training samples (datasets 1-6)
│   └── valid_indices.jsonl  # 1 eval sample (dataset 7)
├── config.yaml              # Training configuration
└── train.py                 # Training script

AgentBench/configs/tasks/
└── os.yaml                  # Task configurations including:
                              # - os-train (datasets 1-6)
                              # - os-eval (dataset 7)
```

## How to Run Training

### Step 1: Start Training Task Server
```bash
cd AgentBench
python -m src.server.task_server_adapter os-train --port 5000
```

### Step 2: Start Evaluation Task Server (in another terminal)
```bash
cd AgentBench
python -m src.server.task_server_adapter os-eval --port 5001
```

### Step 3: Start Training (in another terminal)
```bash
cd AReaL
python -m areal.launcher.local examples/os_interaction/train.py \
    --config examples/os_interaction/config.yaml \
    experiment_name=os_rl \
    trial_name=run1
```

## Configuration Details

### Key Config Settings (config.yaml)
- `max_tokens_per_mb: 8192` - Increased to handle long conversations with system prompts
- `train_dataset.batch_size: 8` - Batch size for training
- `total_train_epochs: 10` - Number of training epochs
- `allocation_mode: sglang:d2p1t1+d2p1t1` - 2 SGLang servers, 2 training processes

### Task Server URLs
- Training: `http://localhost:5000` (os-train task)
- Evaluation: `http://localhost:5001` (os-eval task)

**Note**: You'll need to update `train.py` to use different ports for training and evaluation workflows if running both servers simultaneously.

## Index File Format

Each line in the index files follows this format:
```json
{"index": "train-001-stock-00000", "messages": []}
```

Where:
- `train-001-stock-00000` format: `{split}-{dataset_num}-{problem_name}-{index}`
- `messages`: Empty array (populated during rollout)