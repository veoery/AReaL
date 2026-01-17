# AlfWorld + AReaL Integration Troubleshooting Guide

This document describes issues encountered when integrating AlfWorld with AReaL via the task_server_adapter, and the solutions implemented.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         AReaL Training                           │
│  ┌────────────────────┐         ┌─────────────────────────┐     │
│  │ TaskServerWorkflow │ ◄─HTTP─►│ task_server_adapter.py  │     │
│  │  (rollout worker)  │         │  (AgentBench/FastAPI)   │     │
│  └────────────────────┘         └───────────┬─────────────┘     │
│                                              │                   │
│                                              ▼                   │
│                                    ┌──────────────────┐          │
│                                    │  alfworld/task.py│          │
│                                    │  (async + threads)│         │
│                                    └────────┬─────────┘          │
│                                             │                    │
│                                             ▼                    │
│                                    ┌──────────────────┐          │
│                                    │ TextWorld Engine │          │
│                                    │ (AlfWorld env)   │          │
│                                    └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **AReaL**: RL training framework with async rollout collection
- **task_server_adapter**: HTTP server adapting AgentBench tasks to standard API
- **alfworld/task.py**: AgentBench task implementation using TextWorld
- **TextWorld**: Game engine powering AlfWorld environments

## Problem 1: Task Server Timeouts After 15-20 Turns

### Symptoms
```
20260114-18:43:05.947 TaskServerWorkflow INFO: Running turn 3 / 35
20260114-18:43:05.947 TaskServerWorkflow INFO: action: {'type': 'text', 'content': 'ACTION: use desklamp 1<|im_end|>'}
20260114-18:44:35.951 TaskServerWorkflow ERROR: Episode failed for sample 39: Server returned 500: {"detail":"Task did not respond within 60s"}
```

Training would hang at random turns (usually 15-20) with 60-90s timeouts, then fail with HTTP 500 errors.

### Root Cause

**Issue #1: Synchronous `env.step()` blocking async event loop**

The AgentBench task code called `env.step()` synchronously within an async function:

```python
# BEFORE (BROKEN)
async def alfworld_run(self, session: Session, env: Any):
    for i in range(0, self.max_step):
        output = await session.action()
        action = process_action(output, admissible_commands)

        # BLOCKING CALL - freezes entire event loop!
        observation, reward, done, info = env.step([action])
```

**Why this breaks:**
- `env.step()` is synchronous and can take several seconds (TextWorld game engine)
- When called in an async function, it **blocks the entire event loop**
- All other concurrent episodes freeze waiting for this one `env.step()` to complete
- After 60-90s of queued operations → timeout

**Analogy:**
```
Single-lane bridge:
  Episode 1: [crossing bridge - takes 5 seconds]
  Episode 2:    [waiting... waiting... waiting...]
  Episode 3:       [waiting... waiting... waiting...]
  Episode 4:          [waiting... waiting... timeout!]
```

### Solution #1: Thread Pool Executor

Wrap blocking operations in a thread pool so they don't block the event loop:

```python
# AFTER (FIXED) - in AgentBench/src/server/tasks/alfworld/task.py

import asyncio
import concurrent.futures
import threading

# Thread pool for running blocking env operations
_env_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="alfworld_env"
)

async def alfworld_run(self, session: Session, env: Any):
    loop = asyncio.get_event_loop()

    # env.reset() - run in thread pool
    def safe_env_reset():
        with _env_lock:  # See Problem 3
            return env.reset()

    ob, info = await asyncio.wait_for(
        loop.run_in_executor(_env_executor, safe_env_reset),
        timeout=60.0
    )

    # env.step() - run in thread pool
    for i in range(0, self.max_step):
        output = await session.action()
        action = process_action(output, admissible_commands)

        def safe_env_step():
            with _env_lock:  # See Problem 3
                return env.step([action])

        step_result = await asyncio.wait_for(
            loop.run_in_executor(_env_executor, safe_env_step),
            timeout=30.0
        )
        observation, reward, done, info = step_result
```

**What this achieves:**
- ✅ Event loop stays free - other episodes continue running
- ✅ Timeouts work correctly (30s per step)
- ✅ Up to 16 concurrent env operations (controlled by `max_workers`)

**Files modified:**
- `AgentBench/src/server/tasks/alfworld/task.py` (lines 1-21, 165-242)

---

## Problem 2: Early Task Termination Causes Hangs

### Symptoms
```
# In AgentBench logs:
repeat actions for 3 times: failure
INFO: 10.100.0.1:35694 - "POST /api/episode/step HTTP/1.1" 500 Internal Server Error

# In AReaL logs (90 seconds later):
TaskServerWorkflow ERROR: Episode failed for sample 44: Server returned 500: {"detail":"Task did not respond within 90s"}
```

Tasks that terminated early (repeat action check, max steps, errors) would hang for 90s before failing.

### Root Cause

**Issue #2: task_server_adapter doesn't handle early task completion**

The `step_episode()` function in `task_server_adapter.py` waited only for `agent_pull()` (next observation):

```python
# BEFORE (BROKEN)
async def step_episode(self, episode_id: str, action: Dict[str, Any]):
    episode = self.episodes[episode_id]
    agent_output = AgentOutput(status=AgentOutputStatus.NORMAL, content=action_content)

    # Wait for next observation
    task_output = await asyncio.wait_for(
        episode.session.controller.agent_pull(agent_output),
        timeout=90.0
    )
    # ... process task_output
```

**What happens when task returns early:**
1. Task hits repeat action check → returns with `AGENT_INVALID_ACTION`
2. Task exits without calling `session.controller.agent_signal.release()`
3. `agent_pull()` waits forever for semaphore that will never be released
4. After 90s → timeout → HTTP 500 error

**Architecture of the problem:**

```python
# AgentBench session controller uses semaphores
class SessionController:
    async def agent_pull(self, env_input=None):
        async with self.agent_lock:
            if env_input is not None:
                self.env_input = env_input
                self.env_signal.release()
            await self.agent_signal.acquire()  # ← BLOCKS HERE!
            return self.env_output

# Task returns early
async def alfworld_run(...):
    if repeat_actions_detected:
        return 0, log_info, SampleStatus.AGENT_INVALID_ACTION
        # Never reaches env_pull() → agent_signal never released!
```

### Solution #2: Monitor Task Completion

Use `asyncio.wait()` to monitor **both** the `agent_pull()` call and the task itself:

```python
# AFTER (FIXED) - in AgentBench/src/server/task_server_adapter.py

async def step_episode(self, episode_id: str, action: Dict[str, Any]):
    episode = self.episodes[episode_id]
    agent_output = AgentOutput(status=AgentOutputStatus.NORMAL, content=action_content)

    # Create tasks for both operations
    agent_pull_coro = episode.session.controller.agent_pull(agent_output)
    agent_pull_task = asyncio.create_task(agent_pull_coro)

    # Wait for EITHER agent_pull OR task completion
    done_tasks, pending_tasks = await asyncio.wait(
        [agent_pull_task, episode.task_handle],
        timeout=90.0,
        return_when=asyncio.FIRST_COMPLETED
    )

    # Check what completed
    if episode.task_handle in done_tasks:
        # Task finished early (e.g., repeat action check, max steps, error)
        agent_pull_task.cancel()

        task_result = episode.task_handle.result()
        print(f"Task finished early with status: {task_result.status}")

        # Mark episode as done and return final result immediately
        episode.done = True
        reward = float(task_result.result.get("result", 0.0)) if task_result.result else 0.0

        del self.episodes[episode_id]

        return EpisodeResponse(
            episode_id=episode_id,
            observation=None,
            reward=reward,
            done=True,
            info={"turn": episode.turn + 1, "status": str(task_result.status)}
        )

    elif agent_pull_task in done_tasks:
        # Normal case: agent_pull completed
        task_output = agent_pull_task.result()
        # ... continue normal processing
```

**Applied to both:**
- `start_episode()` - handles task failures during `env.reset()`
- `step_episode()` - handles task early returns during episode

**Files modified:**
- `AgentBench/src/server/task_server_adapter.py` (lines 215-267, 292-405)

---

## Problem 3: TextWorld Parser Thread-Safety Issues

### Symptoms
```
Traceback (most recent call last):
  File "/usr/local/lib/python3.9/site-packages/tatsu/contexts.py", line 538, in _call
    self._rule_stack.pop()
IndexError: pop from empty list

During handling of the above exception:
  File "/usr/local/lib/python3.9/site-packages/textworld/logic/parser.py", line 671, in _pddlStart_
    self._pddlDocument_()
  ...
  File "/usr/local/lib/python3.9/site-packages/tatsu/buffering.py", line 173, in posline
    return self._line_cache[pos].line
IndexError: list index out of range
```

Random parser corruption errors when multiple episodes initialized concurrently.

### Root Cause

**Issue #3: TextWorld's parsers use global state**

TextWorld has multiple parsers that are **not thread-safe**:
1. **PDDL parser** (used in `env.reset()`) - parses game logic
2. **Text generator parser** (used in `env.step()`) - generates observations

When multiple threads call these parsers concurrently, they corrupt each other's state:

```python
# Thread 1: parser.parse(game_file_1)
#   → parser internal state = [stack for game 1]

# Thread 2: parser.parse(game_file_2)  (runs concurrently)
#   → parser internal state = [stack for game 2]  (overwrites Thread 1's state!)

# Thread 1: tries to pop from stack
#   → IndexError: pop from empty list (Thread 2 corrupted it!)
```

**Why this happens:**
- Parser instances use module-level global state (buffers, stacks)
- Python's GIL doesn't prevent this corruption (race condition happens between Python bytecode ops)
- Running in thread pool with `max_workers=16` means up to 16 threads trying to parse simultaneously

### Solution #3: Serialize Environment Operations with Lock

Add a threading lock to ensure only **one thread** calls TextWorld parsers at a time:

```python
# AFTER (FIXED) - in AgentBench/src/server/tasks/alfworld/task.py

import threading

# Lock to prevent concurrent environment operations
# TextWorld's parsers (PDDL, text generator) are NOT thread-safe
# All env.reset() and env.step() calls must be serialized
_env_lock = threading.Lock()

async def alfworld_run(self, session: Session, env: Any):
    loop = asyncio.get_event_loop()

    # Wrap env.reset() with lock
    def safe_env_reset():
        with _env_lock:  # Only ONE thread enters at a time
            return env.reset()

    ob, info = await asyncio.wait_for(
        loop.run_in_executor(_env_executor, safe_env_reset),
        timeout=60.0
    )

    # Wrap env.step() with lock
    for i in range(self.max_step):
        # ... get action ...

        def safe_env_step():
            with _env_lock:  # Only ONE thread enters at a time
                return env.step([action])

        step_result = await asyncio.wait_for(
            loop.run_in_executor(_env_executor, safe_env_step),
            timeout=30.0
        )
```

**What the lock does:**

```
Timeline with lock:
==================
Episode 1: [waiting...] [env.step - has lock] [released lock]
Episode 2: [waiting...] [waiting for lock...] [env.step - has lock] [released]
Episode 3: [waiting...] [waiting for lock...] [waiting...] [env.step - has lock]

Only ONE thread calls TextWorld parsers at a time → no corruption!
```

**Thread lock visualization:**

```python
_env_lock = threading.Lock()

# 16 episodes want to call env.step() simultaneously
# But lock ensures only ONE runs at a time:

Thread 1: with _env_lock:     # Acquires lock
              env.step()      # Runs safely
                              # Releases lock automatically

Thread 2:     [waiting...]    # Tries to acquire lock
          with _env_lock:     # Now acquires lock
              env.step()      # Runs safely
                              # Releases lock

Thread 3:     [waiting...]
          [waiting...]
          with _env_lock:     # Now acquires lock
              env.step()
```

**Performance trade-off:**
- ✅ Stability: No parser corruption
- ⚠️ Serialization: Only 1 env operation at a time (sequential, not parallel)
- ✅ Event loop: Still non-blocking (runs in thread pool)
- ✅ Concurrency: Multiple episodes still run concurrently (model inference happens in parallel)

**Files modified:**
- `AgentBench/src/server/tasks/alfworld/task.py` (lines 18-21, 166-178, 229-242)

---

## Summary of All Changes

### Modified Files

#### 1. `AgentBench/src/server/tasks/alfworld/task.py`

**Imports and globals:**
```python
import asyncio
import concurrent.futures
import threading

_env_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="alfworld_env")
_env_lock = threading.Lock()
```

**Key changes:**
- Wrapped `env.reset()` in `run_in_executor()` with lock (lines 166-178)
- Wrapped `env.step()` in `run_in_executor()` with lock (lines 229-242)
- Added timeouts: 60s for reset, 30s for step

#### 2. `AgentBench/src/server/task_server_adapter.py`

**Key changes:**
- Modified `start_episode()` to detect early task failures (lines 215-267)
- Modified `step_episode()` to detect early task completion (lines 292-405)
- Both now use `asyncio.wait()` to monitor task handle + agent_pull simultaneously

### Configuration Recommendations

**AReaL config (`examples/alfworld/config.yaml`):**

```yaml
rollout:
  max_concurrent_rollouts: 4-16  # Start with 4, can increase to 16

gconfig:
  n_samples: 4  # GRPO: multiple samples per prompt
  max_new_tokens: 256
  temperature: 1.0

actor:
  group_size: ${gconfig.n_samples}  # Must match n_samples for GRPO
  reward_norm:
    mean_level: group    # GRPO: group-level normalization
    std_level: group
    group_size: ${gconfig.n_samples}
  kl_ctl: 0.0           # GRPO typically uses no KL penalty
```

**Performance considerations:**
- `max_concurrent_rollouts`: Controls how many episodes run concurrently
  - Higher = more parallel model inference
  - But env operations are serialized (due to `_env_lock`)
- Thread pool size: Set to 16 in `task.py` (can adjust if needed)
- Recommended: Start with `max_concurrent_rollouts: 4`, monitor performance

---

## How to Run

### 1. Start task server (Terminal 1)
```bash
cd AgentBench
python -m src.server.task_server_adapter alfworld-std --port 5000 \
    --config "configs/tasks/alfworld.yaml"
```

### 2. Start training (Terminal 2)
```bash
cd AReaL
python -m areal.launcher.local examples/alfworld/train.py \
    --config examples/alfworld/config.yaml \
    experiment_name=alfworld_rl \
    trial_name=run1
```

---

## Common Issues

### Issue: Still getting timeouts
- Check if both fixes are applied (thread pool + early termination handling)
- Increase timeout in config if env is genuinely slow: `timeout=120.0`
- Check task server logs for actual errors

### Issue: Parser corruption errors persist
- Ensure `_env_lock` is used in BOTH `env.reset()` AND `env.step()`
- Check no other code path calls env methods without lock
- Verify only one task server instance is running

### Issue: Training is slow
- Env operations are serialized - this is expected
- To improve: reduce `max_concurrent_rollouts` to reduce lock contention
- Model inference still runs in parallel, which is the main bottleneck

### Issue: Memory issues
- Each episode creates a TextWorld environment (memory intensive)
- Reduce `max_concurrent_rollouts` if OOM occurs
- Monitor memory with `htop` or `nvidia-smi`

---

## Technical Deep Dive: Why Standard AgentBench Doesn't Have These Issues

**Standard AgentBench architecture:**
```
python -m src.start_task  →  Spawns multiple worker PROCESSES
                              Each process: 1 task, concurrency=1-5

Process 1 (port 5001): Task instance, can block safely
Process 2 (port 5002): Task instance, independent
Process 3 (port 5003): Task instance, independent
```

- **True parallelism** via separate processes
- If one `env.step()` blocks, only that process is affected
- Each process has its own memory space → no parser corruption

**task_server_adapter architecture (our case):**
```
Single process, single event loop, multiple episodes via async/await

Episode 1 ─┐
Episode 2 ─┼─► Single async event loop
Episode 3 ─┘       ↓
                Thread pool (16 workers)
                   ↓
              TextWorld (shared memory)
```

- **Cooperative concurrency** via async/await
- Originally: blocking `env.step()` froze event loop
- Fixed: thread pool prevents event loop blocking
- But: shared memory requires locks for thread-safety

**Why we need task_server_adapter:**
- AReaL expects a simple HTTP API (start_episode, step_episode)
- Standard AgentBench uses complex controller/worker architecture
- task_server_adapter bridges the gap with a stateless HTTP server

---

## Future Improvements

### Option 1: Process Pool for True Parallelism
Replace `ThreadPoolExecutor` with `ProcessPoolExecutor`:
- Each env runs in separate process → no parser corruption
- No lock needed → full parallelism
- Trade-off: Higher memory usage, slower startup

### Option 2: Cached Environment Initialization
Pre-initialize environments and reuse them:
- Avoid repeated parser loading
- Trade-off: More complex lifecycle management

### Option 3: Native AReaL Integration
Directly integrate with AgentBench's controller/worker:
- Use standard multi-process architecture
- Trade-off: More invasive changes to AReaL

---

## References

- AReaL Documentation: `/areal/README.md`
- AgentBench: https://github.com/THUDM/AgentBench
- TextWorld: https://github.com/microsoft/TextWorld
- AlfWorld: https://github.com/alfworld/alfworld
- Python asyncio: https://docs.python.org/3/library/asyncio.html
- Threading locks: https://docs.python.org/3/library/threading.html#lock-objects