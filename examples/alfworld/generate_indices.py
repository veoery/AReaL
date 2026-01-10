#!/usr/bin/env python3
"""Generate index files for alfworld-std (train) and alfworld-dev (eval) tasks."""

import json
import os

AGENTBENCH_ROOT = "/home/haizhonz/yizhuod/AgentBench"
OUTPUT_DIR = "/home/haizhonz/yizhuod/AReaL/examples/alfworld/data"

def count_games_in_split(split_name):
    """Count how many game files are in a split (dev or standard)."""
    data_path = os.path.join(AGENTBENCH_ROOT, "data/alfworld", f"{split_name}.json")

    with open(data_path) as f:
        content = json.load(f)

    # Content is a dict with task types as keys, lists of file paths as values
    # e.g., {"pick_and_place": ["path1", "path2"], "clean": [...], ...}
    total_games = 0
    for task_type, game_files in content.items():
        total_games += len(game_files)
        print(f"  {task_type}: {len(game_files)} games")

    return total_games

def generate_indices_for_split(split_name, prefix):
    """Generate indices for a split."""
    num_games = count_games_in_split(split_name)

    indices = []
    for i in range(num_games):
        index = f"{prefix}{i:05d}"
        indices.append({"index": index, "messages": []})

    return indices

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training indices (alfworld-std: standard split)
    print("=" * 60)
    print("Generating TRAINING indices (alfworld-std: standard split)")
    print("=" * 60)
    train_indices = generate_indices_for_split("standard", "std-")

    train_output = os.path.join(OUTPUT_DIR, "train_indices.jsonl")
    with open(train_output, 'w') as f:
        for item in train_indices:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Generated {len(train_indices)} training indices -> {train_output}")

    # Generate validation indices (alfworld-dev: dev split)
    print("\n" + "=" * 60)
    print("Generating VALIDATION indices (alfworld-dev: dev split)")
    print("=" * 60)
    valid_indices = generate_indices_for_split("dev", "dev-")

    valid_output = os.path.join(OUTPUT_DIR, "valid_indices.jsonl")
    with open(valid_output, 'w') as f:
        for item in valid_indices:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Generated {len(valid_indices)} validation indices -> {valid_output}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training samples (alfworld-std, standard split): {len(train_indices)}")
    print(f"Validation samples (alfworld-dev, dev split):    {len(valid_indices)}")
    print(f"\nTotal samples: {len(train_indices) + len(valid_indices)}")

if __name__ == "__main__":
    main()
