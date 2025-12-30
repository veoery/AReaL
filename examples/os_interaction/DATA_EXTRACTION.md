# How OS Interaction Data is Extracted and Used

## Overview

Each JSON file contains **multiple problems** in an array. AgentBench automatically:
1. Expands glob patterns to find all JSON files
2. Extracts the filename and incorporates it into the index
3. Loads each problem from the array and assigns a unique index

## Data Structure Example

### stock.json (7 problems)
```json
[
    {  // Problem 0
        "description": "Tell me how many times Alice sold a stock.",
        "create": { "init": {...} },
        "evaluation": { "check": [...] }
    },
    {  // Problem 1
        "description": "Tell me how many times Bob bought a stock.",
        "create": { "init": {...} },
        "evaluation": { "check": [...] }
    },
    {  // Problem 2
        "description": "Count the total number of stocks that Alice bought",
        ...
    },
    // ... 4 more problems (total 7)
]
```

## Complete Index Generation Process

### Step 1: Task Configuration
```yaml
# In os.yaml
os-train:
  data_config:
    files:
      - problem_file: data/os_interaction/data/1/*.json
        script_dir: data/os_interaction/scripts/1/
        index_prefix: "train-001-"
```

### Step 2: Glob Expansion ([task.py:289-304](../../AgentBench/src/server/tasks/os_interaction/task.py#L289-L304))

```python
# Expand glob pattern
glob.glob("data/os_interaction/data/1/*.json")
# Returns: ["data/os_interaction/data/1/stock.json"]

# Build index prefix with filename
filename = os.path.basename("data/os_interaction/data/1/stock.json")  # "stock.json"
filename_no_ext = filename.removesuffix(".json")  # "stock"
full_prefix = "train-001-" + filename_no_ext + "-"
# Result: "train-001-stock-"
```

### Step 3: Load Problems from File ([task.py:306-316](../../AgentBench/src/server/tasks/os_interaction/task.py#L306-L316))

```python
# Load all problems from stock.json
configs = self._load_configs("data/os_interaction/data/1/stock.json")
# Returns list of 7 JudgeConfig objects (one per problem in the array)

# Assign indices to each problem
for idx, config in enumerate(configs):
    index_key = "train-001-stock-" + "%05d" % idx
    problem_configs[index_key] = {
        "file": "data/os_interaction/data/1/stock.json",
        "config": config,  # The actual problem definition
        "index": idx       # Position in the array (0-6)
    }
```

**Generated indices**:
```
train-001-stock-00000  -> stock.json[0] (Alice sold stock)
train-001-stock-00001  -> stock.json[1] (Bob bought stock)
train-001-stock-00002  -> stock.json[2] (Alice total stocks)
train-001-stock-00003  -> stock.json[3] (Bob stock types sold)
train-001-stock-00004  -> stock.json[4] (Bob sold but never bought)
train-001-stock-00005  -> stock.json[5] (Most active trader)
train-001-stock-00006  -> stock.json[6] (Highest transaction stock)
```

## Complete Dataset Breakdown

Based on the actual file structure, here's how many problems each dataset contributes:

### Dataset 1: stock.json
- **7 problems** from stock.json
- Indices: `train-001-stock-00000` through `train-001-stock-00006`

### Dataset 2: environment.json
- **Need to check** - likely multiple problems
- Indices: `train-002-environment-XXXXX`

### Dataset 3: ac.json
- **Need to check** - likely multiple problems
- Indices: `train-003-ac-XXXXX`

### Dataset 4: Multiple files
- N11.json, N225.json, N37.json, N4.json, N41.json
- Q09.json, Q19.json, Q30.json, Q47.json, Q49.json
- **Each file likely contains multiple problems**
- Indices: `train-004-N11-XXXXX`, `train-004-N225-XXXXX`, etc.

### Dataset 5: new.json
- **Need to check** - likely multiple problems
- Indices: `train-005-new-XXXXX`

### Dataset 6: new.json
- **Need to check** - likely multiple problems
- Indices: `train-006-new-XXXXX`

### Dataset 7 (Eval): bootstrap.json
- **Need to check** - likely multiple problems
- Indices: `eval-007-bootstrap-XXXXX`

## How Training Uses the Data

During training, the workflow:

1. **Reads index from train_indices.jsonl**
   ```json
   {"index": "train-001-stock-00000", "messages": []}
   ```

2. **Calls task server with sample_id**
   ```python
   response = await http_client.post(
       "http://localhost:5000/api/episode/start",
       json={"sample_id": "train-001-stock-00000", "config": {}}
   )
   ```

3. **Task server looks up the problem**
   ```python
   problem_config = self.problem_configs["train-001-stock-00000"]
   # Gets: {
   #   "file": "data/os_interaction/data/1/stock.json",
   #   "config": <JudgeConfig for problem 0>,
   #   "index": 0
   # }
   ```

4. **Task server initializes Docker environment**
   - Runs init scripts from `config.create.init`
   - Creates isolated container with the problem setup

5. **Task server sends initial observation**
   - System prompt + one-shot example + problem description
   - Agent sees: "Stock logs are shown in /usr/stock.log..."

6. **Multi-turn interaction**
   - Agent sends bash commands
   - Task executes in Docker and returns output
   - Continues until agent answers or max_turns reached

7. **Evaluation**
   - Compares agent's answer with expected result
   - Uses `evaluation.check` scripts or `evaluation.match` regex
   - Returns reward (1.0 for correct, 0.0 for incorrect)

## Why Our Current Index Files Are Wrong

Our current `train_indices.jsonl` has:
```json
{"index": "train-001-stock-00000", "messages": []}
{"index": "train-002-environment-00000", "messages": []}
...
```

**This assumes**:
- stock.json has only 1 problem (uses index 00000)
- environment.json has only 1 problem (uses index 00000)

**Reality**:
- stock.json has **7 problems** (should have indices 00000-00006)
- Other files likely have multiple problems too!

## Solution: Correct Index Generation

We need to:
1. Count actual problems in each JSON file
2. Generate indices for ALL problems in each file
3. This will give us the true total number of training samples

Let me check how many problems are actually in each file...
