"""Visual diagnostics for collected LatentWAV episodes."""

from __future__ import annotations

from io import BytesIO

import numpy as np

from .loop import EpisodeRollout


def rollout_gif_bytes(rollout: EpisodeRollout, fps=10) -> bytes:
    """Render an episode as an annotated animated GIF."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    from PIL import Image, ImageDraw

    frames = []
    for step, pixels in enumerate(rollout.pixels):
        image = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(image)
        action = rollout.actions[step]
        text = (
            f"step {step:03d}  macro {rollout.macro_step_indices[step]:02d}\n"
            f"a=({action[0]:+.3f}, {action[1]:+.3f})\n"
            f"reward={rollout.rewards[step]:.3f}  "
            f"wav={rollout.disagreements[step]:.4f}"
        )
        box = draw.multiline_textbbox((0, 0), text, spacing=2)
        box_width = box[2] - box[0] + 10
        box_height = box[3] - box[1] + 8
        draw.rectangle((0, 0, box_width, box_height), fill=(255, 255, 255))
        draw.multiline_text((5, 4), text, fill=(0, 0, 0), spacing=2)
        frames.append(image)

    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    return output.getvalue()


def paired_control_gif_bytes(
    expert_frames,
    fine_tuned_frames,
    goal_frame,
    expert_distances,
    fine_tuned_distances,
    expert_actions,
    fine_tuned_actions,
    *,
    replan_steps=(),
    fps=10,
) -> bytes:
    """Render expert, fine-tuned, and goal frames as a side-by-side GIF."""
    if fps <= 0:
        raise ValueError("fps must be positive")

    from PIL import Image, ImageDraw

    expert_frames = np.asarray(expert_frames)
    fine_tuned_frames = np.asarray(fine_tuned_frames)
    goal_frame = np.asarray(goal_frame)
    expert_distances = np.asarray(expert_distances)
    fine_tuned_distances = np.asarray(fine_tuned_distances)
    expert_actions = np.asarray(expert_actions)
    fine_tuned_actions = np.asarray(fine_tuned_actions)

    if expert_frames.shape != fine_tuned_frames.shape:
        raise ValueError("Expert and fine-tuned videos must have matching shapes")
    if expert_frames.ndim != 4 or expert_frames.shape[-1] != 3:
        raise ValueError("Video frames must have shape (T,H,W,3)")
    if goal_frame.shape != expert_frames.shape[1:]:
        raise ValueError("Goal frame must match the rollout frame shape")
    frame_count = len(expert_frames)
    if len(expert_distances) != frame_count or len(fine_tuned_distances) != frame_count:
        raise ValueError("Every video frame needs a goal distance")
    if len(expert_actions) != frame_count - 1 or len(fine_tuned_actions) != frame_count - 1:
        raise ValueError("A T-frame rollout must contain T-1 executed actions")

    height, width = expert_frames.shape[1:3]
    header_height = 42
    replan_steps = set(replan_steps)
    rendered = []
    for step in range(frame_count):
        canvas = Image.new("RGB", (width * 3, height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        is_replan = step in replan_steps
        suffix = "  [CEM REPLAN]" if is_replan else ""

        if step == 0:
            expert_action = "start"
            tuned_action = "start"
        else:
            expert_action = (
                f"a=({expert_actions[step - 1, 0]:+.2f},"
                f"{expert_actions[step - 1, 1]:+.2f})"
            )
            tuned_action = (
                f"a=({fine_tuned_actions[step - 1, 0]:+.2f},"
                f"{fine_tuned_actions[step - 1, 1]:+.2f})"
            )

        labels = (
            f"EXPERT  step {step:02d}{suffix}\nd={expert_distances[step]:.2f}  {expert_action}",
            f"FINE-TUNED  step {step:02d}{suffix}\nd={fine_tuned_distances[step]:.2f}  {tuned_action}",
            "GOAL (same for both)",
        )
        panels = (expert_frames[step], fine_tuned_frames[step], goal_frame)
        for panel_index, (pixels, label) in enumerate(zip(panels, labels)):
            x = panel_index * width
            draw.multiline_text((x + 5, 4), label, fill=(0, 0, 0), spacing=2)
            canvas.paste(Image.fromarray(pixels).convert("RGB"), (x, header_height))
            if panel_index:
                draw.line((x, 0, x, height + header_height), fill=(40, 40, 40), width=2)
        rendered.append(canvas)

    output = BytesIO()
    rendered[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    return output.getvalue()


def rollout_metrics_png_bytes(rollout: EpisodeRollout) -> bytes:
    """Plot reward, WAV disagreement, and executed actions over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(len(rollout))
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, rollout.rewards, color="tab:green")
    axes[0].set_ylabel("reward")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, rollout.disagreements, color="tab:red")
    axes[1].set_ylabel("WAV score")
    axes[1].grid(alpha=0.25)

    for action_dimension in range(rollout.actions.shape[1]):
        axes[2].plot(
            steps,
            rollout.actions[:, action_dimension],
            label=f"action[{action_dimension}]",
        )
    axes[2].set_ylabel("action")
    axes[2].set_xlabel("environment step")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.suptitle("LatentWAV PushT rollout diagnostics")
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150)
    plt.close(figure)
    return output.getvalue()
