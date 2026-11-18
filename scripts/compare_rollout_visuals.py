"""Compare two matched LatentWAV rollout files and render paired visuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-episodes", type=int, default=3)
    parser.add_argument("--baseline-label", default="expert")
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args()


def episode_slice(dataset, episode):
    offset = int(dataset["ep_offset"][episode])
    length = int(dataset["ep_len"][episode])
    return slice(offset, offset + length)


def annotate(frame, label, step, reward, disagreement):
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    text = (
        f"{label}  step {step:03d}\n"
        f"reward={reward:.3f}  wav={disagreement:.4f}"
    )
    bounds = draw.multiline_textbbox((0, 0), text, spacing=2)
    draw.rectangle(
        (0, 0, bounds[2] - bounds[0] + 10, bounds[3] - bounds[1] + 8),
        fill=(255, 255, 255),
    )
    draw.multiline_text((5, 4), text, fill=(0, 0, 0), spacing=2)
    return image


def render_gif(
    output, baseline, candidate, episode, baseline_label, candidate_label
):
    baseline_rows = episode_slice(baseline, episode)
    candidate_rows = episode_slice(candidate, episode)
    length = min(
        int(baseline["ep_len"][episode]), int(candidate["ep_len"][episode])
    )
    frames = []
    for step in range(length):
        baseline_index = baseline_rows.start + step
        candidate_index = candidate_rows.start + step
        left = annotate(
            baseline["pixels"][baseline_index],
            baseline_label,
            step,
            float(baseline["reward"][baseline_index]),
            float(baseline["disagreement"][baseline_index]),
        )
        right = annotate(
            candidate["pixels"][candidate_index],
            candidate_label,
            step,
            float(candidate["reward"][candidate_index]),
            float(candidate["disagreement"][candidate_index]),
        )
        combined = Image.new("RGB", (left.width + right.width, left.height))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        frames.append(combined)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.baseline, "r") as baseline, h5py.File(
        args.candidate, "r"
    ) as candidate:
        episodes = min(len(baseline["ep_len"]), len(candidate["ep_len"]))
        rows = []
        for episode in range(episodes):
            baseline_rows = episode_slice(baseline, episode)
            candidate_rows = episode_slice(candidate, episode)
            length = min(
                int(baseline["ep_len"][episode]),
                int(candidate["ep_len"][episode]),
            )
            baseline_pixels = baseline["pixels"][baseline_rows][:length].astype(
                np.float32
            )
            candidate_pixels = candidate["pixels"][candidate_rows][:length].astype(
                np.float32
            )
            baseline_actions = baseline["action"][baseline_rows][:length]
            candidate_actions = candidate["action"][candidate_rows][:length]
            baseline_rewards = baseline["reward"][baseline_rows][:length]
            candidate_rewards = candidate["reward"][candidate_rows][:length]
            baseline_disagreement = baseline["disagreement"][baseline_rows][:length]
            candidate_disagreement = candidate["disagreement"][candidate_rows][:length]
            rows.append(
                {
                    "episode": episode,
                    "seed": 3407 + episode,
                    "baseline_mean_reward": float(baseline_rewards.mean()),
                    "candidate_mean_reward": float(candidate_rewards.mean()),
                    "reward_change": float(
                        candidate_rewards.mean() - baseline_rewards.mean()
                    ),
                    "baseline_mean_disagreement": float(
                        baseline_disagreement.mean()
                    ),
                    "candidate_mean_disagreement": float(
                        candidate_disagreement.mean()
                    ),
                    "mean_frame_mse": float(
                        np.mean((candidate_pixels - baseline_pixels) ** 2)
                    ),
                    "mean_action_l2": float(
                        np.linalg.norm(
                            candidate_actions - baseline_actions, axis=-1
                        ).mean()
                    ),
                }
            )

        summary = {
            "episodes": episodes,
            "baseline_mean_reward": float(
                np.mean([row["baseline_mean_reward"] for row in rows])
            ),
            "candidate_mean_reward": float(
                np.mean([row["candidate_mean_reward"] for row in rows])
            ),
            "baseline_mean_disagreement": float(
                np.mean([row["baseline_mean_disagreement"] for row in rows])
            ),
            "candidate_mean_disagreement": float(
                np.mean([row["candidate_mean_disagreement"] for row in rows])
            ),
            "per_episode": rows,
        }
        (args.output_dir / "comparison.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        ranked = sorted(rows, key=lambda row: row["mean_frame_mse"], reverse=True)
        for row in ranked[: args.top_episodes]:
            render_gif(
                args.output_dir / f"episode_{row['episode']:03d}_side_by_side.gif",
                baseline,
                candidate,
                row["episode"],
                args.baseline_label,
                args.candidate_label,
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
