"""
A2C training for multi-UAV collaborative search (baseline comparison to PPO).
Parameter sharing is achieved naturally by flattening 3 agents' obs/actions.
Usage:
    python train_a2c.py --timesteps 1000000
"""

import argparse
import os

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from search_env import DroneSearchEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--tb-dir", type=str, default="tb_logs_a2c")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = DroneSearchEnv()
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tb_dir,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        seed=args.seed,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        },
    )

    ckpt = CheckpointCallback(
        save_freq=50_000,
        save_path=args.save_dir,
        name_prefix="a2c_search",
    )

    print(f"[INFO] Starting A2C training for {args.timesteps} timesteps ...")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)

    final_path = os.path.join(args.save_dir, "a2c_search_final")
    model.save(final_path)
    print(f"[INFO] Saved final A2C model to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
