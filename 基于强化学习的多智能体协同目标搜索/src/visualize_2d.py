"""
2D top-down visualization for multi-UAV search with view radius and signal annotation.
Generates animated GIF showing search coverage, obstacles, and signal gradients.
"""

import argparse
import os
import json
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import matplotlib.patheffects as pe


def load_trajectory(csv_path: str):
    import csv
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            aid = int(row["agent_id"])
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            if aid not in data:
                data[aid] = []
            data[aid].append((step, pos))
    # sort by step
    for aid in data:
        data[aid].sort(key=lambda x: x[0])
    return data


def plot_2d_gif(csv_path: str, world_json: str, out_gif: str, fps: int = 2):
    trajs = load_trajectory(csv_path)

    with open(world_json, "r") as f:
        ws = json.load(f)
    space_size = ws["space_size"]
    target_pos = np.array(ws["target_pos"])
    obstacles = ws["obstacles"]

    colors = ["tab:blue", "tab:orange", "tab:green"]
    view_radius = 5.0
    max_len = max(len(v) for v in trajs.values())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, space_size + 1)
    ax.set_ylim(-1, space_size + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Multi-UAV Blind Search – 2D Top-Down View", fontsize=14, fontweight="bold")

    # draw space boundary
    ax.plot([0, space_size, space_size, 0, 0], [0, 0, space_size, space_size, 0],
            "k-", linewidth=2, alpha=0.6)

    # draw obstacles
    for obs in obstacles:
        c = np.array(obs["center"])
        s = np.array(obs["size"])
        rect = Rectangle((c[0] - s[0]/2, c[1] - s[1]/2), s[0], s[1],
                         facecolor="gray", edgecolor="black", alpha=0.35, linewidth=1.5)
        ax.add_patch(rect)
        # label
        ax.text(c[0], c[1], "OBS", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    # target
    ax.scatter(target_pos[0], target_pos[1], c="red", marker="*", s=400,
               zorder=10, edgecolors="darkred", linewidths=1.5, label="Target")

    # view radius circles (will update per frame)
    circles = []
    for aid in range(len(trajs)):
        circ = Circle((0, 0), view_radius, color=colors[aid], alpha=0.08, fill=True, zorder=1)
        ax.add_patch(circ)
        circles.append(circ)

    # trajectory lines with z-height encoded by alpha/linewidth
    lines = []
    for aid in range(len(trajs)):
        l, = ax.plot([], [], color=colors[aid], linewidth=2.5, alpha=0.7, zorder=4)
        lines.append(l)

    # z-height annotations (small text near head)
    z_texts = []
    for aid in range(len(trajs)):
        zt = ax.text(0, 0, "", fontsize=8, color=colors[aid], fontweight="bold",
                     ha="center", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        z_texts.append(zt)

    # agent heads
    heads = []
    for aid in range(len(trajs)):
        h = ax.scatter([], [], c=colors[aid], s=200, marker="o",
                       edgecolors="k", linewidths=2, zorder=6)
        heads.append(h)

    # visited trail heatmap points
    visited_scatters = []
    for aid in range(len(trajs)):
        vs = ax.scatter([], [], c=colors[aid], s=30, alpha=0.25, zorder=2)
        visited_scatters.append(vs)

    # signal strength annotations (top-right panel-like text)
    signal_texts = []
    for aid in range(len(trajs)):
        t = ax.text(0.98, 0.98 - aid*0.06, "", transform=ax.transAxes,
                    color=colors[aid], fontsize=11, fontweight="bold",
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor=colors[aid]))
        signal_texts.append(t)

    # step counter + legend box
    step_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                        fontsize=13, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

    # legend for agents
    legend_handles = [
        plt.Line2D([0], [0], color=colors[i], marker="o", markersize=10, linewidth=2, label=f"UAV {i}")
        for i in range(len(trajs))
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="red", markersize=15, linewidth=0, label="Target"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    def sig_str(pos_3d):
        d = np.linalg.norm(pos_3d - target_pos)
        return np.exp(-d / 8.0)

    def update(frame):
        for aid, pts in trajs.items():
            idx = min(frame, len(pts) - 1)
            step, pos = pts[idx]
            positions = np.array([p[1][:2] for p in pts[:idx+1]])
            z_vals = np.array([p[1][2] for p in pts[:idx+1]])

            # update trajectory line (thicker when high Z, thinner when low Z)
            # use varying alpha to show Z height
            alphas = np.clip(z_vals / space_size, 0.3, 1.0)
            # For line plots we can't vary alpha per segment easily in matplotlib,
            # so we use a solid line with alpha based on average Z
            avg_z = np.mean(z_vals) if len(z_vals) > 0 else 0
            lines[aid].set_data(positions[:, 0], positions[:, 1])
            lines[aid].set_alpha(0.4 + 0.6 * (avg_z / space_size))

            # update head position
            heads[aid].set_offsets([[pos[0], pos[1]]])

            # update view radius circle
            circles[aid].center = (pos[0], pos[1])

            # update visited trail
            visited_scatters[aid].set_offsets(positions)

            # update signal text
            s = sig_str(pos)
            signal_texts[aid].set_text(f"UAV {aid}: signal={s:.3f}")

            # update Z height text (show actual Z to explain obstacle avoidance)
            z_texts[aid].set_position((pos[0], pos[1]))
            z_texts[aid].set_text(f"Z={pos[2]:.1f}")

        step_text.set_text(f"Step {frame+1}/{max_len}")
        if frame + 1 == max_len:
            step_text.set_text(f"Step {frame+1}/{max_len}  ✓ TARGET FOUND!")
            step_text.set_bbox(dict(boxstyle="round,pad=0.5", facecolor="lime", alpha=0.8))

        return lines + heads + circles + visited_scatters + signal_texts + z_texts + [step_text]

    ani = FuncAnimation(fig, update, frames=max_len, interval=1000//fps, blit=False)
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    ani.save(out_gif, writer="pillow", fps=fps)
    print(f"[VIS] Saved 2D GIF: {out_gif}")
    plt.close()

    # also generate a static final-frame figure
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_xlim(-1, space_size + 1)
    ax2.set_ylim(-1, space_size + 1)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (m)", fontsize=12)
    ax2.set_ylabel("Y (m)", fontsize=12)
    ax2.set_title("Multi-UAV Blind Search – Final Coverage Map", fontsize=14, fontweight="bold")

    ax2.plot([0, space_size, space_size, 0, 0], [0, 0, space_size, space_size, 0],
             "k-", linewidth=2, alpha=0.6)

    for obs in obstacles:
        c = np.array(obs["center"])
        s = np.array(obs["size"])
        rect = Rectangle((c[0] - s[0]/2, c[1] - s[1]/2), s[0], s[1],
                         facecolor="gray", edgecolor="black", alpha=0.35, linewidth=1.5)
        ax2.add_patch(rect)

    ax2.scatter(target_pos[0], target_pos[1], c="red", marker="*", s=400,
                zorder=10, edgecolors="darkred", linewidths=1.5, label="Target")

    for aid, pts in trajs.items():
        positions = np.array([p[1][:2] for p in pts])
        ax2.plot(positions[:, 0], positions[:, 1], color=colors[aid], linewidth=2.5, alpha=0.7, zorder=4)
        ax2.scatter(positions[:, 0], positions[:, 1], c=colors[aid], s=30, alpha=0.3, zorder=2)
        ax2.scatter(positions[-1, 0], positions[-1, 1], c=colors[aid], s=300, marker="o",
                    edgecolors="k", linewidths=2, zorder=6, label=f"UAV {aid}")
        # draw final view radius
        circ = Circle((positions[-1, 0], positions[-1, 1]), view_radius,
                      color=colors[aid], alpha=0.12, fill=True, zorder=1)
        ax2.add_patch(circ)

    ax2.legend(loc="lower right", fontsize=10)
    out_png = out_gif.replace(".gif", ".png")
    plt.savefig(out_png, dpi=200)
    print(f"[VIS] Saved 2D PNG: {out_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--world", type=str, required=True)
    parser.add_argument("--out", type=str, default="assets/trajectories_2d/search_2d")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()
    plot_2d_gif(args.csv, args.world, args.out + ".gif", fps=args.fps)


if __name__ == "__main__":
    main()
