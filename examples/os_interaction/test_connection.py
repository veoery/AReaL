"""
Test script to verify task server connection and API.

This script tests the connection to the task server and demonstrates
the API usage.

Usage:
    python examples/os_interaction/test_connection.py --server http://localhost:5000
"""

import argparse
import asyncio
import aiohttp


async def test_task_server(base_url: str):
    """Test task server API."""
    print(f"\n{'='*70}")
    print(f"Testing Task Server: {base_url}")
    print(f"{'='*70}\n")

    async with aiohttp.ClientSession() as session:
        # Test 1: Get task info
        print("Test 1: Get Task Info")
        print("-" * 70)
        try:
            async with session.get(f"{base_url}/api/task/info") as resp:
                if resp.status == 200:
                    info = await resp.json()
                    print(f"✓ Task Name: {info['name']}")
                    print(f"✓ Num Samples: {info['num_samples']}")
                    print(f"✓ Max Episode Length: {info['max_episode_length']}")
                    print(f"✓ Connected successfully!\n")
                else:
                    print(f"✗ Failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

        # Test 2: Start episode
        print("Test 2: Start Episode")
        print("-" * 70)
        # Use actual sample ID from AgentBench dev.json
        sample_id = "dev-001-dev-00000"  # First sample from dev.json
        try:
            async with session.post(
                f"{base_url}/api/episode/start",
                json={"sample_id": sample_id, "config": {}}
            ) as resp:
                if resp.status == 200:
                    start_resp = await resp.json()
                    episode_id = start_resp["episode_id"]
                    observation = start_resp["observation"]
                    print(f"✓ Episode ID: {episode_id}")
                    print(f"✓ Observation Type: {observation['type']}")
                    print(f"✓ Initial Prompt (first 500 chars):")
                    print(f"  {observation['content'][:500]}...")
                    print()
                else:
                    print(f"✗ Failed: {resp.status}")
                    error = await resp.text()
                    print(f"  Error: {error}")
                    return False
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False

        # Test 3: Send action
        print("Test 3: Send Action")
        print("-" * 70)
        test_action = """Think: I need to list files in /etc to count them.

Act: bash

```bash
ls -1 /etc | wc -l
```"""
        test_action = "xyzydsfhis"
        try:
            async with session.post(
                f"{base_url}/api/episode/step",
                json={
                    "episode_id": episode_id,
                    "action": {"type": "text", "content": test_action}
                }
            ) as resp:
                if resp.status == 200:
                    step_resp = await resp.json()
                    observation = step_resp.get("observation")
                    reward = step_resp.get("reward", 0.0)
                    done = step_resp["done"]
                    print(f"✓ Action sent successfully")
                    if observation:
                        print(f"✓ Observation: {observation['content'][:100]}...")
                    print(f"✓ Reward: {reward}")
                    print(f"✓ Done: {done}")
                    print()
                else:
                    print(f"✗ Failed: {resp.status}")
                    error = await resp.text()
                    print(f"  Error: {error}")
                    return False
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False

        # Test 4: Cancel episode
        print("Test 4: Cancel Episode")
        print("-" * 70)
        try:
            async with session.post(
                f"{base_url}/api/episode/cancel",
                json={"episode_id": episode_id}
            ) as resp:
                if resp.status == 200:
                    cancel_resp = await resp.json()
                    print(f"✓ Episode cancelled: {cancel_resp['status']}")
                    print()
                else:
                    print(f"✗ Failed: {resp.status}")
                    error = await resp.text()
                    print(f"  Error: {error}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            # Not critical, continue

        # Test 5: Health check
        print("Test 5: Health Check")
        print("-" * 70)
        try:
            async with session.get(f"{base_url}/api/health") as resp:
                if resp.status == 200:
                    health = await resp.json()
                    print(f"✓ Status: {health['status']}")
                    print(f"✓ Active Episodes: {health['active_episodes']}")
                    print(f"✓ Task Name: {health['task_name']}")
                    print()
                else:
                    print(f"✗ Failed: {resp.status}")
        except Exception as e:
            print(f"✗ Failed: {e}")

    print(f"{'='*70}")
    print("All tests passed! ✓")
    print(f"{'='*70}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test task server connection and API"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:5000",
        help="Task server URL"
    )

    args = parser.parse_args()

    # Run async test
    success = asyncio.run(test_task_server(args.server))

    if success:
        print("✓ Task server is ready for training!")
        print(f"\nTo start training, run:")
        print(f"  python -m areal.launcher.local examples/os_interaction/train.py \\")
        print(f"      --config examples/os_interaction/config.yaml \\")
        print(f"      experiment_name=os_rl \\")
        print(f"      trial_name=test")
    else:
        print("✗ Task server is not ready. Please check the errors above.")
        exit(1)


if __name__ == "__main__":
    main()
