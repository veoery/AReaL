#!/usr/bin/env python3
"""
Quick test to verify the os_interaction dataset can be loaded.
"""

import sys
sys.path.insert(0, '.')

from areal.dataset import get_custom_dataset
from areal.api.cli_args import _DatasetConfig

def test_dataset_loading():
    """Test that os_interaction dataset loads correctly."""

    print("=" * 60)
    print("Testing OS Interaction Dataset Loading")
    print("=" * 60)

    # Test configuration
    config = _DatasetConfig(
        path='examples/os_interaction/data/train_indices.jsonl',
        type='rl',
        max_length=2048
    )

    print(f"\n1. Loading dataset from: {config.path}")
    print(f"   Type: {config.type}")

    try:
        dataset = get_custom_dataset(
            split='train',
            dataset_config=config,
            tokenizer=None,
        )
        print(f"   ✓ Dataset loaded successfully!")

    except Exception as e:
        print(f"   ✗ Failed to load dataset: {e}")
        return False

    # Check dataset size
    print(f"\n2. Dataset size: {len(dataset)} samples")

    # Check first sample structure
    print(f"\n3. First sample structure:")
    first_sample = dataset[0]
    for key in first_sample.keys():
        print(f"   - {key}: {first_sample[key]}")

    # Verify required fields
    print(f"\n4. Verifying required fields:")
    has_index = "index" in first_sample
    has_messages = "messages" in first_sample

    print(f"   - Has 'index' field: {has_index}")
    print(f"   - Has 'messages' field: {has_messages}")

    if has_index and has_messages:
        print(f"\n   ✓ Dataset format is correct!")
        print(f"   Sample index: {first_sample['index']}")
        return True
    else:
        print(f"\n   ✗ Dataset missing required fields")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    print("\n" + "=" * 60)
    if success:
        print("All tests PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    else:
        print("Tests FAILED ✗")
        print("=" * 60)
        sys.exit(1)
