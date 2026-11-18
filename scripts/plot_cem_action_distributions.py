"""Render shared-scale SVG action densities from a CEM comparison report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def plot_report(report_path: Path, output_path: Path) -> None:
    report = json.loads(report_path.read_text())
    if "action_distributions" not in report:
        raise ValueError("Report predates action logging; rerun the CEM comparison")
    comparison = report["action_distributions"]
    x_edges, y_edges = (np.asarray(edge) for edge in comparison["edges"])
    keys = ("expert_train", "expert_cem", "fine_tuned_cem")
    titles = ("IDM expert train", "Expert LeWM CEM", "Fine-tuned LeWM CEM")
    groups = comparison["groups"]
    densities = [
        np.asarray(groups[key]["histogram"], dtype=float) / groups[key]["count"]
        for key in keys
    ]
    positives = np.concatenate([density[density > 0] for density in densities])
    log_min = math.log10(float(positives.min()))
    log_span = max(math.log10(float(positives.max())) - log_min, 1e-12)

    def color(value: float) -> str:
        if value <= 0:
            return "#f1f3f5"
        intensity = (math.log10(value) - log_min) / log_span
        return f"hsl({round(220 - 175 * intensity)},75%,{round(88 - 45 * intensity)}%)"

    width, height, size, top, panel_width = 1180, 440, 300, 78, 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Expert and CEM action density comparison">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="20" y="29" font-family="sans-serif" font-size="20" '
        'fill="#17212b">Executed PushT actions vs IDM training actions</text>',
    ]
    for panel, (key, title, density) in enumerate(zip(keys, titles, densities)):
        left = panel * panel_width + 58
        rows, cols = density.shape
        cell_width, cell_height = size / rows, size / cols
        label = f"{groups[key]['count']:,} actions"
        if key != "expert_train":
            js = comparison["jensen_shannon_bits_vs_expert"][key]
            label += f" · JS {js:.3f} bits"
        parts.extend([
            f'<text x="{left}" y="53" font-family="sans-serif" font-size="16" '
            f'fill="#17212b">{title}</text>',
            f'<text x="{left}" y="69" font-family="sans-serif" font-size="12" '
            f'fill="#59636e">{label}</text>',
        ])
        for ix in range(rows):
            for iy in range(cols):
                x = left + ix * cell_width
                y = top + (cols - iy - 1) * cell_height
                parts.append(
                    f'<rect x="{x:.2f}" y="{y:.2f}" '
                    f'width="{cell_width + 0.05:.2f}" height="{cell_height + 0.05:.2f}" '
                    f'fill="{color(float(density[ix, iy]))}"/>'
                )
        parts.extend([
            f'<rect x="{left}" y="{top}" width="{size}" height="{size}" '
            f'fill="none" stroke="#67737e"/>',
            f'<text x="{left}" y="{top + size + 17}" font-family="sans-serif" '
            f'font-size="11" fill="#59636e">{x_edges[0]:.2f}</text>',
            f'<text x="{left + size}" y="{top + size + 17}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#59636e">{x_edges[-1]:.2f}</text>',
            f'<text x="{left + size / 2}" y="{height - 25}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="#17212b">Relative action x</text>',
        ])
        if panel == 0:
            parts.extend([
                f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11" fill="#59636e">{y_edges[-1]:.2f}</text>',
                f'<text x="{left - 8}" y="{top + size}" text-anchor="end" '
                f'font-family="sans-serif" font-size="11" fill="#59636e">{y_edges[0]:.2f}</text>',
                f'<text x="18" y="{top + size / 2}" '
                f'transform="rotate(-90 18 {top + size / 2})" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13" fill="#17212b">Relative action y</text>',
            ])
    parts.append(
        '<text x="30" y="433" font-family="sans-serif" font-size="11" '
        'fill="#59636e">Shared log-density scale: pale = rare, gold = frequent; gray = empty</text>'
    )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_report(args.report, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
