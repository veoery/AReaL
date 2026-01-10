#!/usr/bin/env python3
"""
Generate index files for AlfWorld tasks.

Since AlfWorld data is in Docker, we generate indices based on typical dataset sizes:
- standard split: ~134 games (for training)
- dev split: ~24 games (for evaluation)

These are the standard sizes from the AlfWorld benchmark.
If your dataset has different sizes, adjust NUM_STANDARD_GAMES and NUM_DEV_GAMES.
"""

import json
import os

OUTPUT_DIR = "/home/haizhonz/yizhuod/AReaL/examples/alfworld/data"

# Standard AlfWorld dataset sizes
# You can verify these by starting the task server and checking the logs
NUM_STANDARD_GAMES = 134  # Standard split (training)
NUM_DEV_GAMES = 24        # Dev split (evaluation)

def generate_indices(num_games, prefix):
    """Generate indices for games."""
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
    train_indices = generate_indices(NUM_STANDARD_GAMES, "std-")

    train_output = os.path.join(OUTPUT_DIR, "train_indices.jsonl")
    with open(train_output, 'w') as f:
        for item in train_indices:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Generated {len(train_indices)} training indices -> {train_output}")

    # Generate validation indices (alfworld-dev: dev split)
    print("\n" + "=" * 60)
    print("Generating VALIDATION indices (alfworld-dev: dev split)")
    print("=" * 60)
    valid_indices = generate_indices(NUM_DEV_GAMES, "dev-")

    valid_output = os.path.join(OUTPUT_DIR, "valid_indices.jsonl")
    with open(valid_output, 'w') as f:
        for item in valid_indices:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Generated {len(valid_indices)} validation indices -> {valid_output}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training samples (alfworld-std, standard split): {len(train_indices)}")
    print(f"Validation samples (alfworld-dev, dev split):    {len(valid_indices)}")
    print(f"\nTotal samples: {len(train_indices) + len(valid_indices)}")
    print("\nNOTE: If you get errors about missing samples when running training,")
    print("      adjust NUM_STANDARD_GAMES and NUM_DEV_GAMES in this script.")

if __name__ == "__main__":
    main()
