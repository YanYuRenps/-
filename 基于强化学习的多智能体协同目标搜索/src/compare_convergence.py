"""
Evaluate PPO vs A2C checkpoints at various timesteps and plot convergence curves.
Usage:
    python compare_convergence.py
"""

import os
import re
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from search_env import DroneSearchEnv


def evaluate_model(model, n_eps=20, max_steps=400):
    env = DroneSearchEnv()
    successes = 0
    steps_list = []
    rewards_list = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        for s in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r
            if done:
                successes += 1
                steps_list.append(s + 1)
                break
            if trunc:
                steps_list.append(max_steps)
                break
        rewards_list.append(ep_reward)
    env.close()
    return {
        "success_rate": successes / n_eps,
        "avg_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "avg_reward": float(np.mean(rewards_list)),
    }


def load_checkpoint(path):
    if "ppo" in path.lower():
        return PPO.load(path)
    else:
        return A2C.load(path)


def extract_steps(name):
    # e.g. ppo_search_150000_steps.zip -> 150000
    # or a2c_search_final.zip -> 1000000
    if "final" in name:
        return 1_000_000
    m = re.search(r"_(\d+)_steps", name)
    if m:
        return int(m.group(1))
    return 0


def main():
    save_dir = "models"
    out_json = "assets/comparison/convergence_data.json"
    out_png = "assets/comparison/ppo_vs_a2c_curve.png"
    os.makedirs("assets/comparison", exist_ok=True)

    # Check if cached results exist
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            results = json.load(f)
        print(f"[INFO] Loaded cached results from {out_json}")
    else:
        results = {"ppo": [], "a2c": []}

        for algo_cls, algo_name in [(PPO, "ppo"), (A2C, "a2c")]:
            pattern = os.path.join(save_dir, f"{algo_name}_search_*.zip")
            paths = sorted(glob(pattern), key=lambda p: extract_steps(os.path.basename(p)))
            print(f"[INFO] Found {len(paths)} {algo_name.upper()} checkpoints")

            for path in paths:
                steps = extract_steps(os.path.basename(path))
                print(f"[EVAL] {algo_name.upper()} @ {steps:,} steps ...", end=" ", flush=True)
                model = algo_cls.load(path)
                metrics = evaluate_model(model, n_eps=20, max_steps=400)
                print(
                    f"success={metrics['success_rate']:.0%} "
                    f"avg_steps={metrics['avg_steps']:.1f} "
                    f"avg_reward={metrics['avg_reward']:.1f}"
                )
                results[algo_name].append({"steps": steps, **metrics})

        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Saved results to {out_json}")

    # Plotting
    ppo_data = sorted(results["ppo"], key=lambda x: x["steps"])
    a2c_data = sorted(results["a2c"], key=lambda x: x["steps"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot 1: Success Rate
    ax = axes[0]
    ax.plot([d["steps"] for d in ppo_data], [d["success_rate"] * 100 for d in ppo_data],
            "o-", color="tab:blue", label="PPO", linewidth=2, markersize=5)
    ax.plot([d["steps"] for d in a2c_data], [d["success_rate"] * 100 for d in a2c_data],
            "s-", color="tab:orange", label="A2C", linewidth=2, markersize=5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Training Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Subplot 2: Average Steps
    ax = axes[1]
    ax.plot([d["steps"] for d in ppo_data], [d["avg_steps"] for d in ppo_data],
            "o-", color="tab:blue", label="PPO", linewidth=2, markersize=5)
    ax.plot([d["steps"] for d in a2c_data], [d["avg_steps"] for d in a2c_data],
            "s-", color="tab:orange", label="A2C", linewidth=2, markersize=5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Steps to Find Target")
    ax.set_title("Search Efficiency vs Training Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Average Reward
    ax = axes[2]
    ax.plot([d["steps"] for d in ppo_data], [d["avg_reward"] for d in ppo_data],
            "o-", color="tab:blue", label="PPO", linewidth=2, markersize=5)
    ax.plot([d["steps"] for d in a2c_data], [d["avg_reward"] for d in a2c_data],
            "s-", color="tab:orange", label="A2C", linewidth=2, markersize=5)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Episode Return")
    ax.set_title("Return vs Training Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("PPO vs A2C Convergence Comparison (20 Fixed Seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=200)
    print(f"[INFO] Saved plot to {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
