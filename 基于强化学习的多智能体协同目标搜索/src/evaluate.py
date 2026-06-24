import numpy as np
from search_env import DroneSearchEnv
from stable_baselines3 import PPO

env = DroneSearchEnv()
model = PPO.load('models/ppo_search_final')

success = 0
steps_list = []
rewards_list = []
for ep in range(20):
    obs, _ = env.reset(seed=ep)
    ep_reward = 0.0
    for s in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        ep_reward += r
        if done:
            success += 1
            steps_list.append(s+1)
            break
        if trunc:
            steps_list.append(400)
            break
    rewards_list.append(ep_reward)

print(f' episodes : 20')
print(f' success  : {success}/20 ({success*5}%)')
print(f' avg_steps: {np.mean(steps_list):.1f}')
print(f' min_steps: {min(steps_list)}')
print(f' avg_reward: {np.mean(rewards_list):.1f}')
