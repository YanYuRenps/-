"""
Decentralized multi-agent PPO training with parameter sharing and cross-agent attention.
Each agent's policy branch strictly processes only its own 40-dim local observation.
Cross-agent attention enables implicit coordination without requiring full joint observation at inference time.

Usage:
    python train_ppo_decentralized.py --timesteps 1000000
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from search_env import DroneSearchEnv


class CrossAgentAttentionExtractor(BaseFeaturesExtractor):
    """
    Parameter-shared feature extractor with scaled dot-product attention across agents.
    Input:  (batch, n_agents * obs_dim) — flattened for SB3 compatibility
    Internally reshaped to (batch, n_agents, obs_dim) and processed independently.
    """

    def __init__(self, observation_space, features_dim: int = 256, n_agents: int = 3, obs_dim: int = 40):
        super().__init__(observation_space, features_dim)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        hidden = 128
        attn_dim = 64

        # Shared encoder: each agent's 40-dim obs -> 128-dim feature (parameter sharing)
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # Cross-agent attention (single-head scaled dot-product)
        self.query = nn.Linear(hidden, attn_dim)
        self.key = nn.Linear(hidden, attn_dim)
        self.value = nn.Linear(hidden, attn_dim)
        self.attn_scale = torch.sqrt(torch.tensor(attn_dim, dtype=torch.float32))

        # Final projection
        total_flat = n_agents * (hidden + attn_dim)
        self.proj = nn.Sequential(
            nn.Linear(total_flat, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        # Decompose joint observation into per-agent local observations
        obs = observations.reshape(batch_size, self.n_agents, self.obs_dim)

        # Parameter-shared encoding: each agent processed by the SAME MLP weights
        encoded = self.shared_encoder(obs)  # (batch, n_agents, hidden)

        # Cross-agent scaled dot-product attention
        Q = self.query(encoded)  # (batch, n_agents, attn_dim)
        K = self.key(encoded)    # (batch, n_agents, attn_dim)
        V = self.value(encoded)  # (batch, n_agents, attn_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attn_scale.to(observations.device)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_agents, n_agents)
        attended = torch.matmul(attn_weights, V)  # (batch, n_agents, attn_dim)

        # Concatenate original encoded features with attended context
        combined = torch.cat([encoded, attended], dim=-1)  # (batch, n_agents, hidden+attn_dim)
        combined = combined.reshape(batch_size, -1)

        return self.proj(combined)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--tb-dir", type=str, default="tb_logs_decentralized")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    n_agents = 3
    env = DroneSearchEnv(n_agents=n_agents)

    policy_kwargs = dict(
        features_extractor_class=CrossAgentAttentionExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_agents=n_agents,
            obs_dim=40,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

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
        policy_kwargs=policy_kwargs,
    )

    ckpt = CheckpointCallback(
        save_freq=50_000,
        save_path=args.save_dir,
        name_prefix="ppo_decentralized",
    )

    print(f"[INFO] Starting decentralized attention training for {args.timesteps} timesteps ...")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=True)

    final_path = os.path.join(args.save_dir, "ppo_decentralized_final")
    model.save(final_path)
    print(f"[INFO] Saved final model to {final_path}")
    env.close()


if __name__ == "__main__":
    main()
