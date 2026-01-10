#!/usr/bin/env python3
"""
Generate numeric index files for AlfWorld.

AlfWorld uses numeric indices (0, 1, 2, ...) not string indices.
This script generates the correct format.
"""

import json
import os

OUTPUT_DIR = "/home/haizhonz/yizhuod/AReaL/examples/alfworld/data"

# Standard AlfWorld dataset sizes
# Adjust these based on your actual dataset
NUM_STANDARD_GAMES = 134  # Standard split (training)
NUM_DEV_GAMES = 24        # Dev split (evaluation)

def generate_indices(num_games, output_file):
    """Generate indices with numeric format."""
    indices = []
    for i in range(num_games):
        # Use integer index, not string!
        indices.append({"index": i, "messages": []})

    with open(output_file, 'w') as f:
        for item in indices:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Generated {len(indices)} numeric indices -> {output_file}")
    print(f"  Indices: {indices[0]['index']} to {indices[-1]['index']}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Generating TRAINING indices (numeric format)")
    print("=" * 70)
    train_output = os.path.join(OUTPUT_DIR, "train_indices.jsonl")
    generate_indices(NUM_STANDARD_GAMES, train_output)

    print("\n" + "=" * 70)
    print("Generating VALIDATION indices (numeric format)")
    print("=" * 70)
    valid_output = os.path.join(OUTPUT_DIR, "valid_indices.jsonl")
    generate_indices(NUM_DEV_GAMES, valid_output)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Training samples:   {NUM_STANDARD_GAMES} (indices 0-{NUM_STANDARD_GAMES-1})")
    print(f"Validation samples: {NUM_DEV_GAMES} (indices 0-{NUM_DEV_GAMES-1})")
    print(f"\nTotal samples: {NUM_STANDARD_GAMES + NUM_DEV_GAMES}")
    print("\nNOTE: AlfWorld uses NUMERIC indices, not string prefixes!")

if __name__ == "__main__":
    main()