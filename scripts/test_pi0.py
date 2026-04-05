#!/usr/bin/env python3
"""Test the fine-tuned Pi0 model on the OMX follower arm.

Usage:
    python scripts/test_pi0.py
    python scripts/test_pi0.py --instruction "pick white pawn from e2, place on e4"
    python scripts/test_pi0.py --model ./local_model --port /dev/tty.usbmodem11401
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.robots.omx_follower.config_omx_follower import OmxFollowerConfig
from lerobot.robots.omx_follower.omx_follower import OmxFollower


FPS = 30
HORIZON = 300  # 10 seconds at 30 fps

# Fields in v0.5.0 config that don't exist in local v0.3.4 PI0Config
_V05_EXTRA_FIELDS = {
    "use_peft", "pretrained_path", "paligemma_variant", "action_expert_variant",
    "dtype", "num_inference_steps", "time_sampling_beta_alpha", "time_sampling_beta_beta",
    "time_sampling_scale", "time_sampling_offset", "min_period", "max_period",
    "rtc_config", "image_resolution", "gradient_checkpointing", "compile_model",
    "compile_mode", "optimizer_grad_clip_norm",
}


def _load_pi0_compat(model_id: str, device: torch.device) -> PI0Policy:
    """Load a Pi0 model trained with LeRobot v0.5.0, stripping incompatible config fields."""
    import json
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    # Download model to local cache
    local_dir = snapshot_download(repo_id=model_id)
    local_dir = Path(local_dir)

    # Patch config.json: remove fields unknown to local PI0Config
    config_path = local_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    patched = False
    for key in _V05_EXTRA_FIELDS:
        if key in cfg:
            del cfg[key]
            patched = True
    if patched:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"       Patched config.json (removed v0.5-only fields)")

    policy = PI0Policy.from_pretrained(str(local_dir))

    # Inject normalization stats from v0.5.0 sidecar files into v0.3.4 buffers
    pre_file = local_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_file = local_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if pre_file.exists() and post_file.exists():
        print("       Injecting normalization stats from sidecar files...")
        pre_stats = load_file(str(pre_file))
        post_stats = load_file(str(post_file))

        # Map sidecar stats → v0.3.4 normalize buffer names
        buf_map = {
            "normalize_inputs.buffer_observation_state.mean": pre_stats["observation.state.mean"],
            "normalize_inputs.buffer_observation_state.std": pre_stats["observation.state.std"],
            "normalize_targets.buffer_action.mean": pre_stats["action.mean"],
            "normalize_targets.buffer_action.std": pre_stats["action.std"],
            "unnormalize_outputs.buffer_action.mean": post_stats["action.mean"],
            "unnormalize_outputs.buffer_action.std": post_stats["action.std"],
        }
        state_dict = policy.state_dict()
        for buf_name, tensor in buf_map.items():
            if buf_name in state_dict:
                state_dict[buf_name].copy_(tensor)
            else:
                # Register as buffer directly
                parts = buf_name.rsplit(".", 1)
                module = policy
                for attr in parts[0].split("."):
                    module = getattr(module, attr)
                module.register_buffer(parts[1], tensor)
        policy.load_state_dict(state_dict, strict=False)
        print("       Normalization stats injected successfully.")

    policy.train(False)
    policy.to(device)
    return policy


def build_observation(robot_obs: dict, instruction: str, device: torch.device) -> dict:
    """Convert raw robot observation into the tensor dict Pi0 expects."""
    # Joint state: 6-DOF float32
    state = torch.from_numpy(
        np.array(
            [robot_obs[k] for k in [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
            ]],
            dtype=np.float32,
        )
    ).unsqueeze(0).to(device)

    # Camera image: (1, C, H, W) float32 normalised to [0, 1]
    img = robot_obs["innomaker"]  # numpy (H, W, 3) uint8
    img_tensor = (
        torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    )

    return {
        "observation.state": state,
        "observation.images.innomaker": img_tensor,
        "task": [instruction],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Pi0 chess model on OMX arm")
    parser.add_argument("--model", default="izchen/pi0_chess", help="HF repo or local path")
    parser.add_argument("--port", default="/dev/tty.usbmodem11401", help="Robot serial port")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--instruction", default=None, help="Language instruction for Pi0")
    parser.add_argument("--steps", type=int, default=HORIZON, help="Max steps to run")
    args = parser.parse_args()

    # --- Connect robot with camera ---
    print(f"[1/3] Connecting robot on {args.port} with camera {args.camera}...")
    robot_config = OmxFollowerConfig(
        port=args.port,
        cameras={
            "innomaker": OpenCVCameraConfig(
                index_or_path=args.camera, fps=FPS, width=1280, height=720,
            ),
        },
    )
    robot = OmxFollower(robot_config)
    robot.connect()

    # --- Load Pi0 policy ---
    print(f"[2/3] Loading Pi0 policy from {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _load_pi0_compat(args.model, device)
    print(f"       Policy loaded on {device}")

    # --- Get instruction ---
    instruction = args.instruction
    if instruction is None:
        instruction = input("[3/3] Enter instruction (e.g. 'pick white pawn from e2, place on e4'): ")
    else:
        print(f"[3/3] Instruction: {instruction}")

    # --- Inference loop ---
    print(f"\nRunning inference for up to {args.steps} steps at {FPS} fps...")
    print("Press Ctrl+C to stop early.\n")

    robot.bus.enable_torque()

    try:
        for step in range(args.steps):
            t0 = time.monotonic()

            # Get observation from robot + camera (retry on transient camera timeout)
            for _attempt in range(3):
                try:
                    obs_raw = robot.get_observation()
                    break
                except TimeoutError:
                    if _attempt < 2:
                        print(f"  [warn] camera timeout at step {step}, retrying...")
                        time.sleep(0.1)
                    else:
                        raise
            obs = build_observation(obs_raw, instruction, device)

            # Run policy
            with torch.inference_mode():
                action = policy.select_action(obs)

            # action may be (n_action_steps, action_dim) or (action_dim,)
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np[0]  # take first step from action chunk
            action_dict = {
                "shoulder_pan.pos": float(action_np[0]),
                "shoulder_lift.pos": float(action_np[1]),
                "elbow_flex.pos": float(action_np[2]),
                "wrist_flex.pos": float(action_np[3]),
                "wrist_roll.pos": float(action_np[4]),
                "gripper.pos": float(action_np[5]),
            }
            robot.send_action(action_dict)

            if step % 30 == 0:
                print(f"  step {step:4d}/{args.steps}  action={[f'{v:.2f}' for v in action_np]}")

            # Maintain target FPS
            elapsed = time.monotonic() - t0
            sleep_time = (1.0 / FPS) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[stopped] Interrupted at step {step}")

    # --- Cleanup ---
    print("Disconnecting robot...")
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
