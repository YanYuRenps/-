"""
Matplotlib 3D visualization for multi-UAV search trajectories.
Generates static PNG and animated GIF (same style as final AUV plots).
Usage:
    python visualize.py --csv data/trajectories/ep_0.csv --out assets/trajectories_3d/search_3d
"""

import argparse
import os
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches


def draw_cube(ax, center, size, color="gray", alpha=0.3):
    """Draw a transparent cube (AABB obstacle)."""
    c = np.array(center)
    s = np.array(size) / 2.0
    verts = [
        [c[0]-s[0], c[1]-s[1], c[2]-s[2]],
        [c[0]+s[0], c[1]-s[1], c[2]-s[2]],
        [c[0]+s[0], c[1]+s[1], c[2]-s[2]],
        [c[0]-s[0], c[1]+s[1], c[2]-s[2]],
        [c[0]-s[0], c[1]-s[1], c[2]+s[2]],
        [c[0]+s[0], c[1]-s[1], c[2]+s[2]],
        [c[0]+s[0], c[1]+s[1], c[2]+s[2]],
        [c[0]-s[0], c[1]+s[1], c[2]+s[2]],
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for e in edges:
        xs = [verts[e[0]][0], verts[e[1]][0]]
        ys = [verts[e[0]][1], verts[e[1]][1]]
        zs = [verts[e[0]][2], verts[e[1]][2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.5)
    # draw semi-transparent faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[0], verts[3], verts[7], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
    ]
    poly3d = Poly3DCollection(faces, alpha=alpha*0.5, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly3d)


def draw_bounding_box(ax, size=20.0, color="black", alpha=0.3):
    """Draw wireframe of the search space."""
    s = size
    corners = [
        [0,0,0], [s,0,0], [s,s,0], [0,s,0],
        [0,0,s], [s,0,s], [s,s,s], [0,s,s]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    for e in edges:
        xs = [corners[e[0]][0], corners[e[1]][0]]
        ys = [corners[e[0]][1], corners[e[1]][1]]
        zs = [corners[e[0]][2], corners[e[1]][2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.0)


def load_trajectory(csv_path: str):
    """Load CSV and return dict of agent_id -> ndarray (N,3)."""
    import csv
    data = {}
    target_found_step = None
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            aid = int(row["agent_id"])
            pos = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            if aid not in data:
                data[aid] = []
            data[aid].append(pos)
            if int(row["found_target"]) == 1 and target_found_step is None:
                target_found_step = step
    for aid in data:
        data[aid] = np.stack(data[aid], axis=0)
    return data, target_found_step


def plot_static_3d(csv_path: str, obstacles: List[Dict], target_pos: np.ndarray,
                   out_png: str, space_size: float = 20.0):
    """Generate static 3D trajectory figure (reference: final AUV fig2_3d_trajectory.png)."""
    trajs, _ = load_trajectory(csv_path)
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # bounding box
    draw_bounding_box(ax, space_size)

    # obstacles
    for obs in obstacles:
        draw_cube(ax, obs["center"], obs["size"], color="gray", alpha=0.25)

    # trajectories
    for aid, pts in trajs.items():
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=colors[aid], linewidth=2.0, label=f"UAV {aid}")
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2],
                   color=colors[aid], marker="o", s=80, edgecolors="k", zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2],
                   color=colors[aid], marker="s", s=80, edgecolors="k", zorder=5)

    # target
    ax.scatter(*target_pos, color="red", marker="*", s=200, label="Target", zorder=6)

    # wind arrow (steady wind indicator)
    W_STEADY = np.array([0.2, 0.1, 0.0])
    wind_origin = np.array([space_size * 0.85, space_size * 0.15, space_size * 0.85])
    wind_scale = 3.0
    ax.quiver(
        wind_origin[0], wind_origin[1], wind_origin[2],
        W_STEADY[0] * wind_scale, W_STEADY[1] * wind_scale, W_STEADY[2] * wind_scale,
        color="darkblue", arrow_length_ratio=0.4, linewidth=2.5, alpha=0.9
    )
    ax.text(
        wind_origin[0] + W_STEADY[0] * wind_scale + 0.3,
        wind_origin[1] + W_STEADY[1] * wind_scale + 0.3,
        wind_origin[2] + W_STEADY[2] * wind_scale + 0.3,
        f"Steady Wind\n{W_STEADY} m/s",
        color="darkblue", fontsize=9, fontweight="bold"
    )

    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Multi-UAV Collaborative Search Trajectories")
    ax.legend(loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"[VIS] Saved static plot: {out_png}")
    plt.close()


def plot_gif_3d(csv_path: str, obstacles: List[Dict], target_pos: np.ndarray,
                out_gif: str, space_size: float = 20.0, fps: int = 2):
    """Generate animated GIF with signal strength annotations and visited trails."""
    trajs, found_step = load_trajectory(csv_path)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    max_len = max(len(v) for v in trajs.values())

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Multi-UAV Blind Search (Signal Gradient)")

    draw_bounding_box(ax, space_size)
    for obs in obstacles:
        draw_cube(ax, obs["center"], obs["size"], color="gray", alpha=0.2)
    ax.scatter(*target_pos, color="red", marker="*", s=200, label="Target", zorder=6)

    # steady wind arrow (static throughout animation)
    W_STEADY = np.array([0.2, 0.1, 0.0])
    wind_origin = np.array([space_size * 0.85, space_size * 0.15, space_size * 0.85])
    wind_scale = 3.0
    ax.quiver(
        wind_origin[0], wind_origin[1], wind_origin[2],
        W_STEADY[0] * wind_scale, W_STEADY[1] * wind_scale, W_STEADY[2] * wind_scale,
        color="darkblue", arrow_length_ratio=0.4, linewidth=2.5, alpha=0.9
    )
    ax.text(
        wind_origin[0] + W_STEADY[0] * wind_scale + 0.3,
        wind_origin[1] + W_STEADY[1] * wind_scale + 0.3,
        wind_origin[2] + W_STEADY[2] * wind_scale + 0.3,
        f"Steady Wind {W_STEADY} m/s",
        color="darkblue", fontsize=9, fontweight="bold"
    )

    lines = []
    heads = []
    for aid in range(len(trajs)):
        l, = ax.plot([], [], [], color=colors[aid], linewidth=2.0, label=f"UAV {aid}")
        h = ax.scatter([], [], [], color=colors[aid], s=80, edgecolors="k", zorder=5)
        lines.append(l)
        heads.append(h)

    # visited trail scatter (all positions visited so far)
    visited_scatters = []
    for aid in range(len(trajs)):
        vs = ax.scatter([], [], [], color=colors[aid], s=15, alpha=0.3, zorder=3)
        visited_scatters.append(vs)

    # signal strength text annotations
    signal_texts = []
    for aid in range(len(trajs)):
        t = ax.text2D(0.02, 0.95 - aid*0.06, "", transform=ax.transAxes,
                      color=colors[aid], fontsize=11, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        signal_texts.append(t)

    # step counter
    step_text = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                          fontsize=12, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.7))

    ax.legend(loc="upper left")

    def sig_str(pos):
        d = np.linalg.norm(pos - target_pos)
        s = np.exp(-d / 8.0)
        return s

    def update(frame):
        for aid, pts in trajs.items():
            idx = min(frame, len(pts) - 1)
            # trajectory line
            lines[aid].set_data(pts[:idx+1, 0], pts[:idx+1, 1])
            lines[aid].set_3d_properties(pts[:idx+1, 2])
            # head
            heads[aid]._offsets3d = ([pts[idx, 0]], [pts[idx, 1]], [pts[idx, 2]])
            # visited trail
            visited_scatters[aid]._offsets3d = (pts[:idx+1, 0].tolist(),
                                                pts[:idx+1, 1].tolist(),
                                                pts[:idx+1, 2].tolist())
            # signal strength annotation
            s = sig_str(pts[idx])
            signal_texts[aid].set_text(f"UAV {aid}: signal={s:.3f}")

        step_text.set_text(f"Step {frame+1}/{max_len}")
        if found_step and frame+1 == found_step:
            step_text.set_text(f"Step {frame+1}/{max_len}  ✓ TARGET FOUND!")
            step_text.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="lime", alpha=0.8))

        return lines + heads + visited_scatters + signal_texts + [step_text]

    ani = FuncAnimation(fig, update, frames=max_len, interval=1000//fps, blit=False)
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    ani.save(out_gif, writer="pillow", fps=fps)
    print(f"[VIS] Saved GIF: {out_gif}")
    plt.close()


import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Trajectory CSV path")
    parser.add_argument("--world", type=str, default=None, help="World state JSON path")
    parser.add_argument("--out", type=str, default="assets/trajectories_3d/search_3d", help="Output prefix")
    parser.add_argument("--target", type=str, default=None, help="Target position x,y,z (override)")
    parser.add_argument("--gif", action="store_true", help="Also generate GIF")
    args = parser.parse_args()

    if args.world and os.path.exists(args.world):
        with open(args.world, "r") as f:
            ws = json.load(f)
        space_size = ws["space_size"]
        target_pos = np.array(ws["target_pos"])
        obstacles = [
            {"center": np.array(o["center"]), "size": np.array(o["size"])}
            for o in ws["obstacles"]
        ]
    else:
        space_size = 15.0
        target_pos = np.array([float(x) for x in args.target.split(",")]) if args.target else np.array([7.5, 7.5, 7.5])
        obstacles = [
            {"center": np.array([5, 5, 5]), "size": np.array([3, 3, 3])},
            {"center": np.array([10, 10, 10]), "size": np.array([3, 3, 3])},
        ]

    plot_static_3d(args.csv, obstacles, target_pos, args.out + ".png", space_size=space_size)
    if args.gif:
        plot_gif_3d(args.csv, obstacles, target_pos, args.out + ".gif", space_size=space_size)


if __name__ == "__main__":
    main()
