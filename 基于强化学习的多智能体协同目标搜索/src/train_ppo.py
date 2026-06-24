"""
PPO training for multi-UAV collaborative search.
Parameter sharing is achieved naturally by flattening 3 agents' obs/actions.
Usage:
    python train_ppo.py --timesteps 300000
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from search_env import DroneSearchEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--tb-dir", type=str, default="tb_logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = DroneSearchEnv()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tb_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
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
        name_prefix="ppo_search",
    )

    print(f"[INFO] Starting training for {args.timesteps} timesteps ...")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)

    final_path = os.path.join(args.save_dir, "ppo_search_final")
    model.save(final_path)
    print(f"[INFO] Saved final model to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
