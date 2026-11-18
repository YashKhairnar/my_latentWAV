"""Compare expert and fine-tuned LeWM with the same online CEM-MPC controller.

The default command is a 10-pair smoke test. Pass ``--episodes 50`` for the
full PushT evaluation size used by the original LeWM protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("latentwav-lewm-control-comparison")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.45.2",
        "hydra-core==1.3.2",
        "h5py",
        "hdf5plugin",
        "numpy",
        "einops",
        "gym==0.26.2",
        "pygame==2.6.1",
        "pymunk==6.6.0",
        "opencv-python-headless",
        "shapely",
        "scikit-image",
        "matplotlib",
        "Pillow",
    )
    .env(
        {
            "PYTHONPATH": "/root/latentwav-src:/root/nano-world-model/src",
            "SDL_VIDEODRIVER": "dummy",
        }
    )
    .add_local_dir("nano-world-model", remote_path="/root/nano-world-model")
    .add_local_dir("src", remote_path="/root/latentwav-src")
)
models_volume = modal.Volume.from_name("latentwav-models")


def _summarize(episodes):
    import numpy as np

    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "mean_goal_state_distance": float(
            np.mean([row["goal_state_distance"] for row in episodes])
        ),
        "mean_block_position_error": float(
            np.mean([row["block_position_error"] for row in episodes])
        ),
        "mean_block_angle_error": float(
            np.mean([row["block_angle_error"] for row in episodes])
        ),
        "mean_final_environment_reward": float(
            np.mean([row["final_environment_reward"] for row in episodes])
        ),
        "mean_initial_frame_mae": float(
            np.mean([row["initial_frame_mae"] for row in episodes])
        ),
    }


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": models_volume},
    timeout=6 * 60 * 60,
    cpu=16,
    memory=65536,
)
def compare(
    episodes: int = 10,
    dataset: str = "/models/datasets/lewm-pusht/pusht_expert_train.h5",
    baseline: str = "/models/checkpoints/lewm-pusht",
    candidate: str = "/models/checkpoints/lewm-wav-20260903-205ep-v2/best",
    seed: int = 3072,
    goal_offset: int = 25,
    eval_budget: int = 50,
    horizon: int = 5,
    execute_blocks: int = 5,
    cem_candidates: int = 300,
    cem_iterations: int = 30,
    cem_elites: int = 30,
    video_episodes: int = 1,
    video_fps: int = 10,
) -> dict:
    """Run paired real-environment rollouts and return a JSON-safe report."""
    import gc

    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch

    from environments.pusht.pusht_env import PushTEnv
    from latentwav.control import CEMConfig, LeWMCEMPlanner
    from latentwav.action_distribution import compare_action_distributions
    from latentwav.data import compute_action_stats, split_episode_ids
    from latentwav.models.world_model import LeWMWorldModel
    from latentwav.visualization import paired_control_gif_bytes

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if goal_offset < 1 or eval_budget < 1:
        raise ValueError("goal_offset and eval_budget must be positive")
    if execute_blocks < 1 or execute_blocks > horizon:
        raise ValueError("execute_blocks must be between 1 and horizon")
    if not 0 <= video_episodes <= episodes:
        raise ValueError("video_episodes must be between 0 and episodes")
    if video_fps < 1:
        raise ValueError("video_fps must be positive")

    models_volume.reload()
    dataset_path = Path(dataset)
    with h5py.File(dataset_path, "r") as source:
        required = {"pixels", "action", "state", "ep_offset", "ep_len"}
        missing = required - set(source.keys())
        if missing:
            raise ValueError(
                f"Online control requires dataset keys {sorted(required)}; "
                f"missing {sorted(missing)}"
            )

        offsets = np.asarray(source["ep_offset"][:], dtype=np.int64)
        lengths = np.asarray(source["ep_len"][:], dtype=np.int64)
        valid_counts = np.maximum(lengths - goal_offset, 0)
        total_pairs = int(valid_counts.sum())
        if episodes > total_pairs:
            raise ValueError(
                f"Requested {episodes} pairs, but only {total_pairs} are valid"
            )

        # Uniform sampling over valid start rows, without materializing them all.
        rng = np.random.default_rng(seed)
        sampled = np.sort(rng.choice(total_pairs, episodes, replace=False))
        cumulative = np.cumsum(valid_counts)
        pair_specs = []
        for pair_id, flat_index in enumerate(sampled):
            episode_id = int(np.searchsorted(cumulative, flat_index, side="right"))
            previous = int(cumulative[episode_id - 1]) if episode_id else 0
            local_start = int(flat_index) - previous
            start_row = int(offsets[episode_id]) + local_start
            goal_row = start_row + goal_offset
            pair_specs.append(
                {
                    "pair_id": pair_id,
                    "source_episode": episode_id,
                    "start_row": start_row,
                    "goal_row": goal_row,
                    "initial_state": np.asarray(source["state"][start_row]).astype(
                        np.float32
                    ),
                    "start_frame": np.asarray(source["pixels"][start_row]),
                    "goal_state": np.asarray(source["state"][goal_row]).astype(
                        np.float32
                    ),
                    "goal_frame": np.asarray(source["pixels"][goal_row]),
                }
            )

        all_actions = np.asarray(source["action"][:], dtype=np.float32)
        valid_actions = all_actions[np.isfinite(all_actions).all(axis=1)]
        action_low = valid_actions.min(axis=0)
        action_high = valid_actions.max(axis=0)
        train_episode_ids, _ = split_episode_ids(len(lengths), seed=0)
        expert_train_actions = [
            all_actions[int(offsets[index]):int(offsets[index] + lengths[index])]
            for index in train_episode_ids
        ]

    action_mean, action_std = compute_action_stats(dataset_path)
    cem_config = CEMConfig(
        horizon=horizon,
        candidates=cem_candidates,
        iterations=cem_iterations,
        elites=cem_elites,
        batch_size=cem_candidates,
    )

    def angle_error(first, second):
        difference = abs(float(first) - float(second)) % (2 * np.pi)
        return min(difference, 2 * np.pi - difference)

    def task_distance(goal, state):
        rotation_error = angle_error(goal[4], state[4])
        difference = np.concatenate((goal[:4] - state[:4], [rotation_error]))
        return float(np.linalg.norm(difference))

    def run_checkpoint(name, checkpoint):
        model = LeWMWorldModel(
            checkpoint,
            device="cuda",
            action_mean=action_mean,
            action_std=action_std,
        )
        planner = LeWMCEMPlanner(model, action_low, action_high, cem_config)
        results = []
        videos = {}

        for pair in pair_specs:
            episode_seed = seed + pair["pair_id"]
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
            with_velocity = len(pair["initial_state"]) >= 7
            environment = PushTEnv(
                render_size=model.image_size,
                relative=True,
                action_scale=100,
                with_velocity=with_velocity,
                with_target=True,
            )
            environment.seed(episode_seed)
            environment.reset_to_state = pair["initial_state"].copy()
            observation, current_state = environment.reset()
            reset_frame = observation["visual"].copy()

            frame_history = [reset_frame]
            action_history = []
            rewards = []
            planning_costs = []
            executed_actions = []
            steps = 0
            replan = 0
            record_video = pair["pair_id"] < video_episodes
            if record_video:
                video_frames = [reset_frame.copy()]
                video_distances = [task_distance(pair["goal_state"], current_state)]
                video_actions = []

            try:
                while steps < eval_budget:
                    plan_seed = seed + pair["pair_id"] * 1000 + replan
                    planned_blocks, predicted_cost = planner.plan(
                        frame_history,
                        action_history,
                        pair["goal_frame"],
                        seed=plan_seed,
                    )
                    planning_costs.append(predicted_cost)

                    for block in planned_blocks[:execute_blocks]:
                        if steps + len(block) > eval_budget:
                            break
                        for action in block:
                            observation, reward, _, info = environment.step(action)
                            executed_actions.append(np.asarray(action, dtype=np.float32).copy())
                            rewards.append(float(reward))
                            current_state = info["state"]
                            steps += 1
                            if record_video:
                                video_frames.append(observation["visual"].copy())
                                video_distances.append(
                                    task_distance(pair["goal_state"], current_state)
                                )
                                video_actions.append(np.asarray(action).copy())

                        action_history.append(np.asarray(block, dtype=np.float32))
                        frame_history.append(observation["visual"])
                        if len(frame_history) > model.history_size:
                            frame_history.pop(0)
                            action_history.pop(0)
                    replan += 1

                goal = pair["goal_state"]
                position_error = float(np.linalg.norm(goal[:4] - current_state[:4]))
                rotation_error = angle_error(goal[4], current_state[4])
                task_difference = np.concatenate(
                    (goal[:4] - current_state[:4], [rotation_error])
                )
                initial_frame_mae = float(
                    np.abs(
                        reset_frame.astype(np.float32)
                        - pair["start_frame"].astype(np.float32)
                    ).mean()
                )
                results.append(
                    {
                        "pair_id": pair["pair_id"],
                        "source_episode": pair["source_episode"],
                        "start_row": pair["start_row"],
                        "goal_row": pair["goal_row"],
                        "steps": steps,
                        "replans": replan,
                        "success": bool(
                            position_error < 20 and rotation_error < np.pi / 9
                        ),
                        "goal_state_distance": float(np.linalg.norm(task_difference)),
                        "agent_position_error": float(
                            np.linalg.norm(goal[:2] - current_state[:2])
                        ),
                        "block_position_error": float(
                            np.linalg.norm(goal[2:4] - current_state[2:4])
                        ),
                        "block_angle_error": rotation_error,
                        "final_environment_reward": rewards[-1] if rewards else 0.0,
                        "max_environment_reward": max(rewards, default=0.0),
                        "mean_predicted_goal_cost": float(np.mean(planning_costs)),
                        "executed_actions": np.asarray(executed_actions).tolist(),
                        "initial_frame_mae": initial_frame_mae,
                    }
                )
                if record_video:
                    videos[pair["pair_id"]] = {
                        "frames": np.asarray(video_frames),
                        "distances": np.asarray(video_distances),
                        "actions": np.asarray(video_actions),
                    }
                print(
                    f"[{name}] pair={pair['pair_id'] + 1}/{len(pair_specs)} "
                    f"success={results[-1]['success']} "
                    f"distance={results[-1]['goal_state_distance']:.2f}"
                )
            finally:
                environment.close()

        del planner, model
        gc.collect()
        torch.cuda.empty_cache()
        return results, videos

    baseline_results, baseline_videos = run_checkpoint("expert", baseline)
    candidate_results, candidate_videos = run_checkpoint("fine_tuned", candidate)
    baseline_summary = _summarize(baseline_results)
    candidate_summary = _summarize(candidate_results)
    distance_changes = [
        candidate_row["goal_state_distance"] - baseline_row["goal_state_distance"]
        for baseline_row, candidate_row in zip(baseline_results, candidate_results)
    ]
    distance_changes_array = np.asarray(distance_changes)
    success_pairs = [
        (baseline_row["success"], candidate_row["success"])
        for baseline_row, candidate_row in zip(baseline_results, candidate_results)
    ]

    report = {
        "evaluation": "paired online goal-conditioned CEM-MPC",
        "goal_distribution": (
            f"future frames {goal_offset} raw steps later in the expert dataset"
        ),
        "dataset": dataset,
        "baseline_checkpoint": baseline,
        "candidate_checkpoint": candidate,
        "config": {
            "seed": seed,
            "episodes": episodes,
            "goal_offset": goal_offset,
            "eval_budget": eval_budget,
            "action_block": 5,
            "horizon": horizon,
            "execute_blocks_before_replan": execute_blocks,
            "cem_candidates": cem_candidates,
            "cem_iterations": cem_iterations,
            "cem_elites": cem_elites,
        },
        "expert": {"summary": baseline_summary, "episodes": baseline_results},
        "fine_tuned": {
            "summary": candidate_summary,
            "episodes": candidate_results,
        },
        "paired_comparison": {
            "success_rate_change": (
                candidate_summary["success_rate"] - baseline_summary["success_rate"]
            ),
            "mean_goal_state_distance_change": float(np.mean(distance_changes)),
            "median_goal_state_distance_change": float(
                np.median(distance_changes_array)
            ),
            "goal_state_distance_change_quartiles": [
                float(value)
                for value in np.quantile(distance_changes_array, [0.25, 0.5, 0.75])
            ],
            "fine_tuned_distance_wins": int(np.sum(distance_changes_array < 0)),
            "expert_distance_wins": int(np.sum(distance_changes_array > 0)),
            "distance_ties": int(np.sum(distance_changes_array == 0)),
            "both_succeed": sum(expert and tuned for expert, tuned in success_pairs),
            "fine_tuned_only_succeeds": sum(
                (not expert) and tuned for expert, tuned in success_pairs
            ),
            "expert_only_succeeds": sum(
                expert and (not tuned) for expert, tuned in success_pairs
            ),
            "neither_succeeds": sum(
                (not expert) and (not tuned) for expert, tuned in success_pairs
            ),
        },
        "action_distributions": compare_action_distributions(
            expert_train_actions,
            [row["executed_actions"] for row in baseline_results],
            [row["executed_actions"] for row in candidate_results],
        ),
    }

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_output = Path(
        f"/models/evaluations/lewm_control_action_comparison_{timestamp}.json"
    )
    report["remote_report_path"] = str(remote_output)
    remote_output.parent.mkdir(parents=True, exist_ok=True)
    remote_output.write_text(json.dumps(report, indent=2) + "\n")
    models_volume.commit()
    print(json.dumps(report["paired_comparison"], indent=2))
    videos = []
    replan_interval = execute_blocks * 5
    replan_steps = range(0, eval_budget, replan_interval)
    for pair_id in range(video_episodes):
        expert_video = baseline_videos[pair_id]
        tuned_video = candidate_videos[pair_id]
        videos.append(
            {
                "pair_id": pair_id,
                "gif": paired_control_gif_bytes(
                    expert_video["frames"],
                    tuned_video["frames"],
                    pair_specs[pair_id]["goal_frame"],
                    expert_video["distances"],
                    tuned_video["distances"],
                    expert_video["actions"],
                    tuned_video["actions"],
                    replan_steps=replan_steps,
                    fps=video_fps,
                ),
            }
        )
    return {"report": report, "videos": videos}


@app.local_entrypoint()
def main(
    episodes: int = 10,
    output: str = "artifacts/lewm_control_action_comparison.json",
    dataset: str = "/models/datasets/lewm-pusht/pusht_expert_train.h5",
    baseline: str = "/models/checkpoints/lewm-pusht",
    candidate: str = "/models/checkpoints/lewm-wav-20260903-205ep-v2/best",
    seed: int = 3072,
    video_episodes: int = 1,
    video_fps: int = 10,
):
    import subprocess
    import sys

    result = compare.remote(
        episodes=episodes,
        dataset=dataset,
        baseline=baseline,
        candidate=candidate,
        seed=seed,
        video_episodes=video_episodes,
        video_fps=video_fps,
    )
    report = result["report"]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    plot_path = output_path.with_name(f"{output_path.stem}_action_distributions.svg")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("plot_cem_action_distributions.py")),
            "--report", str(output_path),
            "--output", str(plot_path),
        ],
        check=True,
    )
    for video in result["videos"]:
        video_path = output_path.with_name(
            f"{output_path.stem}_pair_{video['pair_id']:03d}.gif"
        )
        video_path.write_bytes(video["gif"])
        print(f"Saved paired rollout GIF to {video_path}")
    print(f"Saved paired control report to {output_path}")
