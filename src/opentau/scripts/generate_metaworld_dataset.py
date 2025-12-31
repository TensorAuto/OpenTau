#!/usr/bin/env python3
"""
Script to generate a Metaworld dataset with 100 trajectories of the button-press-v3 task.
Uses SawyerButtonPressV3Policy as the expert policy to collect demonstrations.
Saves the dataset in lerobot format and uploads to the hub.
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import cv2

# Import gymnasium for environment creation and video recording
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from metaworld.policies.policy import Policy
from metaworld.policies.sawyer_assembly_v3_policy import SawyerAssemblyV3Policy
from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.policies.sawyer_bin_picking_v3_policy import SawyerBinPickingV3Policy
from metaworld.policies.sawyer_box_close_v3_policy import SawyerBoxCloseV3Policy
from metaworld.policies.sawyer_button_press_topdown_v3_policy import (
    SawyerButtonPressTopdownV3Policy,
)
from metaworld.policies.sawyer_button_press_topdown_wall_v3_policy import (
    SawyerButtonPressTopdownWallV3Policy,
)

# Import metaworld components
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.policies.sawyer_button_press_wall_v3_policy import (
    SawyerButtonPressWallV3Policy,
)
from metaworld.policies.sawyer_coffee_button_v3_policy import SawyerCoffeeButtonV3Policy
from metaworld.policies.sawyer_coffee_pull_v3_policy import SawyerCoffeePullV3Policy
from metaworld.policies.sawyer_coffee_push_v3_policy import SawyerCoffeePushV3Policy
from metaworld.policies.sawyer_dial_turn_v3_policy import SawyerDialTurnV3Policy
from metaworld.policies.sawyer_disassemble_v3_policy import SawyerDisassembleV3Policy
from metaworld.policies.sawyer_door_close_v3_policy import SawyerDoorCloseV3Policy
from metaworld.policies.sawyer_door_lock_v3_policy import SawyerDoorLockV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from metaworld.policies.sawyer_door_unlock_v3_policy import SawyerDoorUnlockV3Policy
from metaworld.policies.sawyer_drawer_close_v3_policy import SawyerDrawerCloseV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from metaworld.policies.sawyer_faucet_close_v3_policy import SawyerFaucetCloseV3Policy
from metaworld.policies.sawyer_faucet_open_v3_policy import SawyerFaucetOpenV3Policy
from metaworld.policies.sawyer_hammer_v3_policy import SawyerHammerV3Policy
from metaworld.policies.sawyer_hand_insert_v3_policy import SawyerHandInsertV3Policy
from metaworld.policies.sawyer_handle_press_side_v3_policy import (
    SawyerHandlePressSideV3Policy,
)
from metaworld.policies.sawyer_handle_press_v3_policy import SawyerHandlePressV3Policy
from metaworld.policies.sawyer_handle_pull_side_v3_policy import (
    SawyerHandlePullSideV3Policy,
)
from metaworld.policies.sawyer_handle_pull_v3_policy import SawyerHandlePullV3Policy
from metaworld.policies.sawyer_lever_pull_v3_policy import SawyerLeverPullV3Policy
from metaworld.policies.sawyer_peg_insertion_side_v3_policy import (
    SawyerPegInsertionSideV3Policy,
)
from metaworld.policies.sawyer_peg_unplug_side_v3_policy import (
    SawyerPegUnplugSideV3Policy,
)
from metaworld.policies.sawyer_pick_out_of_hole_v3_policy import (
    SawyerPickOutOfHoleV3Policy,
)
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_pick_place_wall_v3_policy import (
    SawyerPickPlaceWallV3Policy,
)
from metaworld.policies.sawyer_plate_slide_back_side_v3_policy import (
    SawyerPlateSlideBackSideV3Policy,
)
from metaworld.policies.sawyer_plate_slide_back_v3_policy import (
    SawyerPlateSlideBackV3Policy,
)
from metaworld.policies.sawyer_plate_slide_side_v3_policy import (
    SawyerPlateSlideSideV3Policy,
)
from metaworld.policies.sawyer_plate_slide_v3_policy import SawyerPlateSlideV3Policy
from metaworld.policies.sawyer_push_back_v3_policy import SawyerPushBackV3Policy
from metaworld.policies.sawyer_push_v3_policy import SawyerPushV3Policy
from metaworld.policies.sawyer_push_wall_v3_policy import SawyerPushWallV3Policy
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from metaworld.policies.sawyer_reach_wall_v3_policy import SawyerReachWallV3Policy
from metaworld.policies.sawyer_shelf_place_v3_policy import SawyerShelfPlaceV3Policy
from metaworld.policies.sawyer_soccer_v3_policy import SawyerSoccerV3Policy
from metaworld.policies.sawyer_stick_pull_v3_policy import SawyerStickPullV3Policy
from metaworld.policies.sawyer_stick_push_v3_policy import SawyerStickPushV3Policy
from metaworld.policies.sawyer_sweep_into_v3_policy import SawyerSweepIntoV3Policy
from metaworld.policies.sawyer_sweep_v3_policy import SawyerSweepV3Policy
from metaworld.policies.sawyer_window_close_v3_policy import SawyerWindowCloseV3Policy
from metaworld.policies.sawyer_window_open_v3_policy import SawyerWindowOpenV3Policy

# Import lerobot components
from src.opentau.datasets.lerobot_dataset import LeRobotDataset
from src.opentau.datasets.utils import DEFAULT_FEATURES
from src.opentau.utils.utils import init_logging

metaworld_task = {
    "assembly-v3": {"task_name": "Pick up a nut and place it onto a peg.", "policy": SawyerAssemblyV3Policy},
    "basketball-v3": {
        "task_name": "Dunk the basketball into the basket.",
        "policy": SawyerBasketballV3Policy,
    },
    "bin-picking-v3": {
        "task_name": "Grasp the puck from one bin and place it into another bin.",
        "policy": SawyerBinPickingV3Policy,
    },
    "box-close-v3": {
        "task_name": "Grasp the cover and close the box with it.",
        "policy": SawyerBoxCloseV3Policy,
    },
    "button-press-topdown-v3": {
        "task_name": "Press a button from the top.",
        "policy": SawyerButtonPressTopdownV3Policy,
    },
    "button-press-topdown-wall-v3": {
        "task_name": "Bypass a wall and press a button from the top.",
        "policy": SawyerButtonPressTopdownWallV3Policy,
    },
    "button-press-v3": {"task_name": "Press a button.", "policy": SawyerButtonPressV3Policy},
    "button-press-wall-v3": {
        "task_name": "Bypass a wall and press a button.",
        "policy": SawyerButtonPressWallV3Policy,
    },
    "coffee-button-v3": {
        "task_name": "Push a button on the coffee machine.",
        "policy": SawyerCoffeeButtonV3Policy,
    },
    "coffee-pull-v3": {"task_name": "Pull a mug from a coffee machine.", "policy": SawyerCoffeePullV3Policy},
    "coffee-push-v3": {"task_name": "Push a mug under a coffee machine.", "policy": SawyerCoffeePushV3Policy},
    "dial-turn-v3": {"task_name": "Rotate a dial 180 degrees.", "policy": SawyerDialTurnV3Policy},
    "disassemble-v3": {"task_name": "Pick a nut out of the a peg.", "policy": SawyerDisassembleV3Policy},
    "door-close-v3": {"task_name": "Close a door with a revolving joint.", "policy": SawyerDoorCloseV3Policy},
    "door-lock-v3": {
        "task_name": "Lock the door by rotating the lock clockwise.",
        "policy": SawyerDoorLockV3Policy,
    },
    "door-open-v3": {"task_name": "Open a door with a revolving joint.", "policy": SawyerDoorOpenV3Policy},
    "door-unlock-v3": {
        "task_name": "Unlock the door by rotating the lock counter-clockwise.",
        "policy": SawyerDoorUnlockV3Policy,
    },
    "hand-insert-v3": {"task_name": "Insert the gripper into a hole.", "policy": SawyerHandInsertV3Policy},
    "drawer-close-v3": {"task_name": "Push and close a drawer.", "policy": SawyerDrawerCloseV3Policy},
    "drawer-open-v3": {"task_name": "Open a drawer.", "policy": SawyerDrawerOpenV3Policy},
    "faucet-open-v3": {
        "task_name": "Rotate the faucet counter-clockwise.",
        "policy": SawyerFaucetOpenV3Policy,
    },
    "faucet-close-v3": {"task_name": "Rotate the faucet clockwise.", "policy": SawyerFaucetCloseV3Policy},
    "hammer-v3": {"task_name": "Hammer a screw on the wall.", "policy": SawyerHammerV3Policy},
    "handle-press-side-v3": {
        "task_name": "Press a handle down sideways.",
        "policy": SawyerHandlePressSideV3Policy,
    },
    "handle-press-v3": {"task_name": "Press a handle down.", "policy": SawyerHandlePressV3Policy},
    "handle-pull-side-v3": {
        "task_name": "Pull a handle up sideways.",
        "policy": SawyerHandlePullSideV3Policy,
    },
    "handle-pull-v3": {"task_name": "Pull a handle up.", "policy": SawyerHandlePullV3Policy},
    "lever-pull-v3": {"task_name": "Pull a lever down 90 degrees.", "policy": SawyerLeverPullV3Policy},
    "pick-place-wall-v3": {
        "task_name": "Pick a puck, bypass a wall and place the puck.",
        "policy": SawyerPickPlaceWallV3Policy,
    },
    "pick-out-of-hole-v3": {
        "task_name": "Pick up a puck from a hole.",
        "policy": SawyerPickOutOfHoleV3Policy,
    },
    "pick-place-v3": {"task_name": "Pick and place a puck to a goal.", "policy": SawyerPickPlaceV3Policy},
    "plate-slide-v3": {"task_name": "Slide a plate into a cabinet.", "policy": SawyerPlateSlideV3Policy},
    "plate-slide-side-v3": {
        "task_name": "Slide a plate into a cabinet sideways.",
        "policy": SawyerPlateSlideSideV3Policy,
    },
    "plate-slide-back-v3": {
        "task_name": "Get a plate from the cabinet.",
        "policy": SawyerPlateSlideBackV3Policy,
    },
    "plate-slide-back-side-v3": {
        "task_name": "Get a plate from the cabinet sideways.",
        "policy": SawyerPlateSlideBackSideV3Policy,
    },
    "peg-insert-side-v3": {"task_name": "Insert a peg sideways.", "policy": SawyerPegInsertionSideV3Policy},
    "peg-unplug-side-v3": {"task_name": "Unplug a peg sideways.", "policy": SawyerPegUnplugSideV3Policy},
    "soccer-v3": {"task_name": "Kick a soccer into the goal.", "policy": SawyerSoccerV3Policy},
    "stick-push-v3": {
        "task_name": "Grasp a stick and push a box using the stick.",
        "policy": SawyerStickPushV3Policy,
    },
    "stick-pull-v3": {
        "task_name": "Grasp a stick and pull a box with the stick.",
        "policy": SawyerStickPullV3Policy,
    },
    "push-v3": {"task_name": "Push the puck to a goal.", "policy": SawyerPushV3Policy},
    "push-wall-v3": {
        "task_name": "Bypass a wall and push a puck to a goal.",
        "policy": SawyerPushWallV3Policy,
    },
    "push-back-v3": {"task_name": "Pull a puck to a goal.", "policy": SawyerPushBackV3Policy},
    "reach-v3": {"task_name": "Reach a goal position.", "policy": SawyerReachV3Policy},
    "reach-wall-v3": {"task_name": "Bypass a wall and reach a goal.", "policy": SawyerReachWallV3Policy},
    "shelf-place-v3": {
        "task_name": "Pick and place a puck onto a shelf.",
        "policy": SawyerShelfPlaceV3Policy,
    },
    "sweep-into-v3": {"task_name": "Sweep a puck into a hole.", "policy": SawyerSweepIntoV3Policy},
    "sweep-v3": {"task_name": "Sweep a puck off the table.", "policy": SawyerSweepV3Policy},
    "window-open-v3": {"task_name": "Push and open a window.", "policy": SawyerWindowOpenV3Policy},
    "window-close-v3": {"task_name": "Push and close a window.", "policy": SawyerWindowCloseV3Policy},
}


def create_metaworld_features() -> Dict[str, Any]:
    """Create feature definitions for metaworld button-press-v3 task."""
    features = DEFAULT_FEATURES.copy()

    # Add image feature (camera observation)
    features["observation.image"] = {
        "dtype": "image",
        "names": ["height", "width", "channels"],
        "shape": (480, 480, 3),  # Actual metaworld image size
    }

    # Add state feature (robot state + goal)
    features["observation.state"] = {
        "dtype": "float32",
        "names": None,
        "shape": (4,),  # Metaworld state dimension for button-press-v3
    }

    # robot state
    features["observation.robot_state"] = {
        "dtype": "float32",
        "names": None,
        "shape": (39,),  # 3-d cartesian position of the end effector and the gripper opening size
    }

    # Add action feature
    features["action"] = {
        "dtype": "float32",
        "names": None,
        "shape": (4,),  # 3-d cartesian position of the end effector and the gripper opening size
    }

    return features


def flip_video_and_delete_original(original_video_path: str) -> str:
    """Flip a video vertically using ffmpeg and delete the original video."""
    original_path = Path(original_video_path)
    flipped_path = original_path.parent / f"{original_path.stem}_flipped{original_path.suffix}"

    # Use FFmpeg to flip the video vertically
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        str(original_path),
        "-vf",
        "vflip",  # Vertical flip filter
        "-c:a",
        "copy",  # Copy audio without re-encoding
        "-y",  # Overwrite output file
        str(flipped_path),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        logging.info(f"Video flipped successfully: {flipped_path}")

        # Delete the original video
        original_path.unlink()
        logging.info(f"Original video deleted: {original_path}")

        return str(flipped_path)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error flipping video: {e}")
        raise


def record_example_video(env, policy: Policy, video_path: str, max_steps: int = 500) -> None:
    """Record a single example video of the expert policy."""
    # Extract folder and prefix from video_path if provided, otherwise use defaults
    video_folder = str(Path(video_path).parent)
    name_prefix = Path(video_path).stem

    # Wrap environment with RecordVideo
    env = RecordVideo(
        env,
        video_folder=video_folder,  # Folder to save videos
        name_prefix=name_prefix,  # Prefix for video filenames
        episode_trigger=lambda x: True,  # Record every episode
    )

    obs_tuple = env.reset()
    obs = obs_tuple[0]  # Extract observation from tuple

    for step in range(max_steps):
        # Get action from expert policy
        action = policy.get_action(obs)

        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)

        # Check if task is completed
        if info["success"] or done or truncated:
            logging.info(
                f"Example video: Task completed at step {step + 1} with reward {1 if info['success'] else 0}"
            )
            break

    env.close()

    # Get the path of the recorded video
    original_video_path = f"{video_folder}/{name_prefix}-episode-0.mp4"
    logging.info(f"Example video saved to: {original_video_path}")

    # Flip the video and delete the original
    flipped_video_path = flip_video_and_delete_original(original_video_path)
    logging.info(f"Flipped video available at: {flipped_video_path}")


def collect_trajectory(env, policy: Policy, task: str, max_steps: int = 500) -> list:
    """Collect a single trajectory using the expert policy."""
    obs_tuple = env.reset()
    obs = obs_tuple[0]  # Extract observation from tuple
    trajectory = []

    for step in range(max_steps):
        # Get action from expert policy
        action = policy.get_action(obs)

        # Render image and flip it vertically
        image = env.render()
        image = cv2.flip(image, 0)

        # Store frame data with proper data types
        assert obs.shape == (39,)
        assert action.shape == (4,)
        frame = {
            "observation.image": image.astype(np.uint8),  # Convert to uint8 for images
            "observation.state": obs.astype(np.float32)[:4],  # Convert to float32
            "observation.robot_state": obs.astype(np.float32),  # Convert to float32
            "action": np.array(action, dtype=np.float32),  # Convert to float32
            "task": task,
        }
        trajectory.append(frame)

        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)

        # Check if task is completed
        if info["success"] or done or truncated:
            logging.info(f"Task completed at step {step + 1} with reward {1 if info['success'] else 0}")
            assert info["success"], "Task was not successful"
            break

    if info["success"] or done:
        return trajectory
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate Metaworld button-press-v3 dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g., 'username/button-press-v3-dataset')",
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=100, help="Number of trajectories to collect (default: 100)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps per trajectory (default: 500)"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for the dataset (default: 30)")
    parser.add_argument(
        "--root", type=str, default=None, help="Root directory to save dataset (default: ./datasets)"
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Upload dataset to Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="Make the dataset repository private")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["metaworld", "button-press", "expert-demos"],
        help="Tags for the dataset",
    )
    parser.add_argument(
        "--record-example-video",
        action="store_true",
        help="Record a single example video of the expert policy",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="all",
        help="task name to be loaded (default: MT-1)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="corner",
        choices=["corner", "corner2", "corner3", "corner4", "topview", "behindGripper", "gripperPOV"],
        help="Camera view for rendering (default: corner)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Initialize logging
    init_logging(level=logging.DEBUG if args.debug else logging.INFO)

    # Set up paths
    if args.root is None:
        root = Path("./datasets") / args.repo_id.replace("/", "_")
    else:
        root = Path(args.root) / args.repo_id.replace("/", "_")

    logging.info(f"Creating dataset with {args.num_trajectories} trajectories")
    logging.info(f"Dataset will be saved to: {root}")
    logging.info(f"Repository ID: {args.repo_id}")

    # Initialize environment and policy
    logging.info("Initializing Metaworld environment and expert policy...")

    # Set camera name from command line argument
    camera_name = args.camera_name

    # Create features
    features = create_metaworld_features()

    # Create dataset
    logging.info("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        features=features,
        use_videos=False,  # Disable videos for now due to encoder issues
        standardize=False,  # Keep raw data for now
    )

    task_list = []
    task_list = list(metaworld_task) if args.task_name == "all" else args.task_name.split(",")
    # Create environment using gym.make approach

    logging.info(f"The list of task to collect data are {task_list}")
    for task_name in task_list:
        logging.info(f"The current task name is {task_name}")
        env = gym.make("Meta-World/MT1", env_name=task_name, render_mode="rgb_array", camera_name=camera_name)
        logging.info(f"Created environment with camera: {camera_name}")

        policy = metaworld_task[task_name]["policy"]()

        # Record example video if requested
        if args.record_example_video:
            logging.info("Recording example video...")
            video_path = root / "example_video"
            record_example_video(env, policy, str(video_path), args.max_steps)
            logging.info("Example video recording completed!")
            return

        # Collect trajectories
        logging.info(f"Collecting {args.num_trajectories} trajectories...")
        successful_trajectories = 0

        for traj_idx in range(args.num_trajectories):
            try:
                logging.info(f"Collecting trajectory {traj_idx + 1}/{args.num_trajectories}")

                # Collect trajectory
                trajectory = collect_trajectory(
                    env, policy, metaworld_task[task_name]["task_name"], args.max_steps
                )

                if len(trajectory) > 0:
                    # Add frames to dataset
                    for frame in trajectory:
                        dataset.add_frame(frame)

                    # Save episode
                    dataset.save_episode()

                    successful_trajectories += 1
                    logging.info(
                        f"Successfully collected trajectory {traj_idx + 1} with {len(trajectory)} steps"
                    )
                else:
                    logging.warning(f"Empty trajectory {traj_idx + 1}, skipping...")

            except Exception as e:
                logging.error(f"Error collecting trajectory {traj_idx + 1}: {e}")
                continue

        logging.info(f"Successfully collected {successful_trajectories}/{args.num_trajectories} trajectories")
        env.close()

    # Upload to hub if requested
    if args.push_to_hub:
        logging.info("Uploading dataset to Hugging Face Hub...")
        try:
            dataset.push_to_hub(tags=args.tags, private=args.private, license="apache-2.0")
            logging.info(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            logging.error(f"Failed to upload dataset: {e}")
            raise
    else:
        logging.info(f"Dataset saved locally at: {root}")
        logging.info("To upload to hub, run with --push-to-hub flag")

    # Print summary
    logging.info("Dataset generation completed!")
    logging.info(f"Total trajectories: {successful_trajectories}")
    logging.info(f"Total frames: {dataset.meta.total_frames}")
    logging.info(f"Dataset features: {list(features.keys())}")


if __name__ == "__main__":
    main()
