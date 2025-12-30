#!/usr/bin/env python3
"""
Generate index files for OS interaction training.

This script creates:
- train_indices.jsonl: All problems from datasets 1-6
- valid_indices.jsonl: All problems from dataset 7
"""

import json
import sys
import os
from pathlib import Path

# Add AgentBench to path
agentbench_path = Path(__file__).parent.parent.parent.parent / "AgentBench"
sys.path.insert(0, str(agentbench_path))

from src.configs import ConfigLoader
from src.typings import InstanceFactory

def get_task_indices(task_name: str, config_path: str = "configs/tasks/os.yaml"):
    """Get all indices from a task."""
    os.chdir(agentbench_path)

    config_loader = ConfigLoader()
    config = config_loader.load_from(config_path)

    if task_name not in config:
        raise ValueError(f"Task '{task_name}' not found in config")

    # Create task instance
    task = InstanceFactory.parse_obj(config[task_name]).create()

    # Get indices
    indices = task.get_indices()

    # Release resources
    task.release()

    return indices

def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("Generating index files for OS interaction training...")
    print(f"Output directory: {output_dir}")

    # Get training indices (datasets 1-6)
    print("\n→ Getting training indices (datasets 1-6)...")
    train_indices = get_task_indices("os-train")
    print(f"  Found {len(train_indices)} training samples")

    # Get eval indices (dataset 7)
    print("\n→ Getting eval indices (dataset 7)...")
    eval_indices = get_task_indices("os-eval")
    print(f"  Found {len(eval_indices)} eval samples")

    # Write train indices
    train_file = output_dir / "train_indices.jsonl"
    print(f"\n→ Writing training indices to {train_file}...")
    with open(train_file, "w") as f:
        for idx in train_indices:
            f.write(json.dumps({"index": idx, "messages": []}) + "\n")
    print(f"  ✓ Wrote {len(train_indices)} training indices")

    # Write eval indices
    eval_file = output_dir / "valid_indices.jsonl"
    print(f"\n→ Writing eval indices to {eval_file}...")
    with open(eval_file, "w") as f:
        for idx in eval_indices:
            f.write(json.dumps({"index": idx, "messages": []}) + "\n")
    print(f"  ✓ Wrote {len(eval_indices)} eval indices")

    print("\n" + "="*80)
    print("Index generation complete!")
    print("="*80)
    print(f"Training samples: {len(train_indices)}")
    print(f"Eval samples: {len(eval_indices)}")
    print(f"Total samples: {len(train_indices) + len(eval_indices)}")
    print("="*80)

if __name__ == "__main__":
    main()
