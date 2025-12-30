"""
Test script for SGLang inference matching AReaL's exact inference process.

This script mimics how AReaL sends requests to SGLang:
1. Takes conversation history (messages)
2. Applies chat template to get input_ids
3. Sends input_ids to SGLang /generate endpoint
4. Decodes the response

Usage:
    python examples/os_interaction/test_sglang_inference.py --base-url http://localhost:10000
"""

import argparse
import asyncio
import aiohttp
import sys
from pathlib import Path
from transformers import AutoTokenizer


# Test conversation histories (what the task server sends to AReaL)
TEST_CASES = [
    {
        "name": "Simple greeting",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    },
    {
        "name": "Simple instruction",
        "messages": [
            {"role": "user", "content": "Count to 5"}
        ]
    },
    {
        "name": "Multi-turn conversation",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
    }
]


async def test_sglang_generation(
    base_url: str,
    tokenizer,
    temperature: float = 0.6,
    max_new_tokens: int = 256
):
    """Test SGLang generation using AReaL's exact request format."""
    print(f"\n{'='*80}")
    print(f"Testing SGLang Generation (AReaL format)")
    print(f"Base URL: {base_url}")
    print(f"Temperature: {temperature}, Max New Tokens: {max_new_tokens}")
    print(f"{'='*80}\n")

    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'-'*80}")
            print(f"Test {i}/{len(TEST_CASES)}: {test_case['name']}")
            print(f"{'-'*80}")

            messages = test_case['messages']

            # Display conversation history
            print("\nConversation History:")
            for msg in messages:
                role = msg['role'].upper()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  [{role}] {content}")

            # Step 1: Apply chat template (exactly like AReaL does)
            print("\n→ Step 1: Apply chat template")
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                print(f"  ✓ Generated {len(input_ids)} input tokens")

                # Show decoded prompt for debugging
                decoded_prompt = tokenizer.decode(input_ids)
                print(f"\n  Decoded prompt (first 500 chars):")
                print(f"  {decoded_prompt[:500]}...")
            except Exception as e:
                print(f"  ✗ Failed to apply chat template: {e}")
                continue

            # Step 2: Build SGLang request (exactly like AReaL does)
            print("\n→ Step 2: Build SGLang request")
            payload = {
                "input_ids": list(input_ids),
                "sampling_params": {
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "frequency_penalty": 0.0,
                },
                "return_logprob": True,
                "stream": False,
            }
            print(f"  ✓ Payload prepared (input_ids: {len(payload['input_ids'])} tokens)")

            # Step 3: Send to SGLang
            print(f"\n→ Step 3: Send to {base_url}/generate")
            try:
                start_time = asyncio.get_event_loop().time()

                async with session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60.0)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"  ✗ Request failed with status {resp.status}")
                        print(f"    Error: {error_text}")
                        continue

                    result = await resp.json()
                    elapsed = asyncio.get_event_loop().time() - start_time

                print(f"  ✓ Response received in {elapsed:.2f}s")
            except asyncio.TimeoutError:
                print(f"  ✗ Request timed out after 60s")
                print(f"    THIS WOULD CAUSE TRAINING TO HANG!")
                continue
            except Exception as e:
                print(f"  ✗ Request failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Step 4: Parse response (exactly like AReaL does)
            print("\n→ Step 4: Parse response")
            try:
                meta_info = result.get("meta_info", {})
                finish_reason = meta_info.get("finish_reason", {})
                stop_reason = finish_reason.get("type", "unknown")

                # Extract output tokens and logprobs (same as SGLangBackend)
                output_token_logprobs = meta_info.get("output_token_logprobs", [])
                output_tokens = [x[1] for x in output_token_logprobs]
                output_logprobs = [x[0] for x in output_token_logprobs]

                print(f"  ✓ Generated {len(output_tokens)} output tokens")
                print(f"  ✓ Stop reason: {stop_reason}")

                # Decode output
                output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

                print(f"\n{'─'*80}")
                print("Generated Output:")
                print(f"{'─'*80}")
                print(output_text)
                print(f"{'─'*80}")

                # Stats
                print(f"\nGeneration Stats:")
                print(f"  Input tokens: {len(input_ids)}")
                print(f"  Output tokens: {len(output_tokens)}")
                print(f"  Total tokens: {len(input_ids) + len(output_tokens)}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Tokens/sec: {len(output_tokens)/elapsed:.1f}")

                # Check for potential issues
                if elapsed > 10.0:
                    print(f"\n  ⚠ WARNING: Generation took {elapsed:.2f}s (> 10s)")
                    print(f"    This might cause timeouts during training!")

                if len(output_tokens) == 0:
                    print(f"\n  ✗ WARNING: No tokens generated!")
                    print(f"    Stop reason: {stop_reason}")

            except Exception as e:
                print(f"  ✗ Failed to parse response: {e}")
                print(f"    Raw response: {result}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*80}")
    print("Testing Complete")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test SGLang inference matching AReaL's exact process"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:10000",
        help="SGLang server base URL (e.g., http://localhost:10000)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for tokenizer (default: infer from config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")

    # Try to infer model path from config if not provided
    if args.model_path is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                args.model_path = config.get("model_path")
                print(f"Using model path from config: {args.model_path}")

    if args.model_path is None:
        print("Error: --model-path required or config.yaml must exist")
        print("Usage: python test_sglang_inference.py --model-path /path/to/model")
        sys.exit(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"✓ Loaded tokenizer from {args.model_path}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)

    # Run async test
    asyncio.run(test_sglang_generation(
        args.base_url,
        tokenizer,
        args.temperature,
        args.max_tokens
    ))


if __name__ == "__main__":
    main()
