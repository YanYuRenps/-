"""
Statistical evaluation script with standard deviations and confidence metrics.
Usage:
    python evaluate_stats.py --model models/ppo_search_final --n-agents 3 --episodes 50 --seed-start 1000
"""

import argparse
import json
import os
import numpy as np
from stable_baselines3 import PPO
from search_env import DroneSearchEnv


def evaluate(model_path, n_agents=3, n_episodes=50, seed_start=1000, deterministic=True):
    env = DroneSearchEnv(n_agents=n_agents)
    model = PPO.load(model_path)

    success_list = []
    steps_list = []
    rewards_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        ep_reward = 0.0
        for s in range(400):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r
            if done:
                success_list.append(1)
                steps_list.append(s + 1)
                break
            if trunc:
                success_list.append(0)
                steps_list.append(400)
                break
        rewards_list.append(ep_reward)

    success_arr = np.array(success_list, dtype=np.float32)
    steps_arr = np.array(steps_list, dtype=np.float32)
    rewards_arr = np.array(rewards_list, dtype=np.float32)

    success_rate = float(np.mean(success_arr))
    # Wilson score interval approximation for binary proportion std
    n = len(success_arr)
    success_std = np.sqrt(success_rate * (1 - success_rate) / n) * 100 if n > 0 else 0.0

    # Filter successful episodes for step statistics
    success_steps = steps_arr[success_arr > 0]

    stats = {
        "episodes": n,
        "success_rate": success_rate,
        "success_std": success_std,
        "steps_mean": float(np.mean(steps_arr)),
        "steps_std": float(np.std(steps_arr)),
        "steps_min": int(np.min(steps_arr)),
        "steps_max": int(np.max(steps_arr)),
        "steps_median": float(np.median(steps_arr)),
        "success_steps_mean": float(np.mean(success_steps)) if len(success_steps) > 0 else float('nan'),
        "success_steps_std": float(np.std(success_steps)) if len(success_steps) > 0 else float('nan'),
        "reward_mean": float(np.mean(rewards_arr)),
        "reward_std": float(np.std(rewards_arr)),
        "reward_min": float(np.min(rewards_arr)),
        "reward_max": float(np.max(rewards_arr)),
    }
    return stats


def print_stats(stats, label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"  Evaluation: {label}")
    print(f"{'='*60}")
    print(f"  Episodes            : {stats['episodes']}")
    print(f"  Success Rate        : {stats['success_rate']*100:.1f}% +/- {stats['success_std']:.1f}%")
    print(f"  Steps (all) mean+/-std : {stats['steps_mean']:.1f} +/- {stats['steps_std']:.1f}")
    print(f"  Steps (all) min/max  : {stats['steps_min']} / {stats['steps_max']}")
    print(f"  Steps (all) median   : {stats['steps_median']:.1f}")
    if not np.isnan(stats['success_steps_mean']):
        print(f"  Steps (success) mean+/-std: {stats['success_steps_mean']:.1f} +/- {stats['success_steps_std']:.1f}")
    print(f"  Reward mean+/-std     : {stats['reward_mean']:.1f} +/- {stats['reward_std']:.1f}")
    print(f"  Reward min/max      : {stats['reward_min']:.1f} / {stats['reward_max']:.1f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model zip file (without .zip)")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--label", type=str, default="", help="Label for printout")
    parser.add_argument("--output", type=str, default="", help="Optional JSON file to save stats")
    args = parser.parse_args()

    stats = evaluate(
        model_path=args.model,
        n_agents=args.n_agents,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=args.deterministic,
    )
    print_stats(stats, label=args.label if args.label else args.model)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Stats saved to {args.output}")


if __name__ == "__main__":
    main()
