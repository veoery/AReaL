"""
Test rollout-only script for OS interaction task.

This script runs ONLY the rollout (interaction with task server) without training.
It mimics what happens during AReaL training but skips the actual model updates.

Usage:
    python examples/os_interaction/test_rollout_only.py \
        --config examples/os_interaction/config.yaml \
        --num-episodes 5
"""

import asyncio
import sys
from pathlib import Path

# Add AReaL to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from areal.api.alloc_mode import AllocationMode
from areal.workflow.task_server import TaskServerWorkflow
from areal.api.cli_args import InferenceEngineConfig, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine

def _init_rollout(
        config: GRPOConfig,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
    ) -> InferenceEngine:
        """
        Initialize rollout engine - EXACTLY matching PPOTrainer._init_rollout().

        This does NOT pass addr to engine.initialize(), allowing the engine to
        auto-discover servers from AREAL_LLM_SERVER_ADDRS environment variable
        (set by the launcher).
        """
        from copy import deepcopy
        allocation_mode = AllocationMode.from_str(config.allocation_mode)
        if allocation_mode.gen_backend == "sglang":
            engine = RemoteSGLangEngine(deepcopy(rollout_config))
        elif allocation_mode.gen_backend == "vllm":
            engine = RemotevLLMEngine(deepcopy(rollout_config))
        else:
            raise ValueError(
                f"Invalid backend: {allocation_mode.gen_backend}, expected sglang or vllm"
            )

        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            engine.config.max_head_offpolicyness = int(1e12)

        # CRITICAL: Do NOT pass addr parameter!
        # Let engine auto-discover from AREAL_LLM_SERVER_ADDRS environment variable
        # This matches exactly what PPOTrainer does in areal/experimental/trainer/rl.py:415
        engine.initialize(train_data_parallel_size=allocation_mode.train.dp_size)
        return engine


async def run_rollout_only(
    config_args: list[str],
):
    """Run rollout-only test without training."""

    # When launched via torchrun, only run on rank 0
    # import os
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # if local_rank != 0:
    #     print(f"Rank {local_rank}: Skipping (only rank 0 runs rollout test)")
    #     return True

    # Load config - build args for load_expr_config
    print("→ Loading configuration...")
    config, _ = load_expr_config(config_args, GRPOConfig)

    # Hard-coded test parameters (can be overridden via Hydra command line)
    task_server_url = "http://localhost:5000"
    num_episodes = 1
    max_turns = 8

    print(f"\n{'='*80}")
    print("Rollout-Only Test (No Training)")
    print(f"{'='*80}")
    print(f"Config file: {config_args[1] if len(config_args) > 1 else 'N/A'}")
    print(f"Task Server: {task_server_url}")
    print(f"Episodes to test: {num_episodes}")
    print(f"Max turns per episode: {max_turns}")
    print(f"{'='*80}\n")

    print(f"  Model: {config.actor.path}")
    print(f"  Experiment: {config.experiment_name}/{config.trial_name}")

    # Load tokenizer (use same utility as train.py)
    print("\n→ Loading tokenizer...")
    from areal.utils.hf_utils import load_hf_tokenizer
    try:
        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        print(f"  ✓ Tokenizer loaded from {config.tokenizer_path}")
    except Exception as e:
        print(f"  ✗ Failed to load tokenizer: {e}")
        sys.exit(1)

    # Generation config is already in config.gconfig
    print("\n→ Generation config:")
    print(f"  Temperature: {config.gconfig.temperature}")
    print(f"  Max new tokens: {config.gconfig.max_new_tokens}")
    print(f"  Top-p: {config.gconfig.top_p}")

    # Setup inference engine (same as train.py)
    print("\n→ Setting up inference engine...")
    print(f"  Engine will auto-discover servers from AREAL_LLM_SERVER_ADDRS environment variable")

    rollout_engine = _init_rollout(
        config,
        config.rollout,
        is_eval=False,
    )
    print(f"  ✓ Rollout engine initialized")

    # Setup workflow (same as train.py)
    print("\n→ Setting up task server workflow...")
    workflow = TaskServerWorkflow(
        task_server_url=task_server_url,
        gconfig=config.gconfig,  # Use config.gconfig directly
        tokenizer=tokenizer,
        max_turns=max_turns,
        turn_discount=0.95,
        rollout_stat_scope="test-rollout",
        timeout=300.0,
    )
    print(f"  ✓ Workflow ready")

    # Load dataset for sample indices
    print("\n→ Loading dataset indices...")
    dataset_path = config.train_dataset.path
    if not dataset_path:
        print("  ✗ Error: train_dataset.path not found in config")
        sys.exit(1)

    import json
    with open(dataset_path) as f:
        indices = [json.loads(line)["index"] for line in f]

    print(f"  ✓ Loaded {len(indices)} sample indices")
    print(f"  Testing with first {num_episodes} samples")

    # Create fake dataloader items (just need index and messages)
    test_samples = [
        {"index": indices[i], "messages": []}
        for i in range(min(num_episodes, len(indices)))
    ]

    # Run rollouts
    print(f"\n{'='*80}")
    print("Starting Rollouts")
    print(f"{'='*80}\n")

    success_count = 0
    failure_count = 0

    for i, sample in enumerate(test_samples, 1):
        sample_idx = sample["index"]

        print(f"\n{'-'*80}")
        print(f"Episode {i}/{len(test_samples)}: {sample_idx}")
        print(f"{'-'*80}")

        try:
            # Run single rollout using the workflow's arun_episode method
            # This is the standard AReaL workflow API
            print(f"→ Starting rollout...")
            result = await workflow.arun_episode(rollout_engine, sample)

            # Check if rollout was rejected (returns None on error)
            if result is None:
                print(f"\n✗ Rollout rejected by workflow (returned None)")
                failure_count += 1
                continue

            # Extract stats from result tensors
            # Result format: {input_ids, logprobs, loss_mask, versions, rewards, attention_mask}
            # All tensors have batch dimension (shape: [1, seq_len] or [1])
            reward_tensor = result.get("rewards")
            if reward_tensor is not None:
                reward = reward_tensor.item()  # Extract scalar from tensor
            else:
                reward = 0.0

            input_ids = result.get("input_ids")
            if input_ids is not None:
                num_tokens = input_ids.shape[1]  # [1, seq_len]
                input_ids_list = input_ids[0].tolist()  # Remove batch dim
            else:
                num_tokens = 0
                input_ids_list = []

            print(f"\n✓ Rollout completed successfully!")
            print(f"  Reward: {reward:.4f}")
            print(f"  Total tokens: {num_tokens}")

            # Show some trajectory info
            if len(input_ids_list) > 0:
                # Decode full trajectory to see what happened
                full_text = tokenizer.decode(input_ids_list, skip_special_tokens=False)
                print(f"\n  Full trajectory (first 1000 chars):")
                print(f"  {'─'*76}")
                print(f"  {full_text[:1000]}...")
                print(f"  {'─'*76}")

            success_count += 1

        except Exception as e:
            print(f"\n✗ Rollout failed!")
            print(f"  Error: {e}")

            import traceback
            print(f"\n  Traceback:")
            traceback.print_exc()

            failure_count += 1

    # Summary
    print(f"\n{'='*80}")
    print("Rollout Test Summary")
    print(f"{'='*80}")
    print(f"Total episodes: {len(test_samples)}")
    print(f"Completed: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Completed rate: {success_count/len(test_samples)*100:.1f}%")
    print(f"{'='*80}\n")

    if failure_count > 0:
        print("⚠ Some rollouts failed. Check errors above.")
        return False
    else:
        print("✓ All rollouts completed successfully!")
        return True


def main():
    """
    Main entry point - compatible with launcher.

    Usage with launcher (auto-starts SGLang servers):
        python -m areal.launcher.local examples/os_interaction/test_rollout_only.py \\
            --config examples/os_interaction/config.yaml \\
            task_server_url=http://localhost:5000 \\
            test_num_episodes=3 \\
            test_max_turns=10

    All arguments are passed to Hydra via load_expr_config().
    No argparse needed - launcher handles everything.
    """
    # Pass all arguments directly to rollout function
    # Launcher will handle SGLang server startup and set AREAL_LLM_SERVER_ADDRS
    success = asyncio.run(run_rollout_only(sys.argv[1:]))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
