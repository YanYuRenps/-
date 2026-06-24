"""
Multi-UAV collaborative 3D target search with steady wind and gust.
Pure NumPy kinematic model. Matplotlib 3D visualization after training.
"""

import os
import csv
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneSearchEnv(gym.Env):
    """
    n_agents UAVs search a 3D space for a hidden target.
    Actions: flattened (n_agents, 3) continuous direction vectors
    Obs: flattened (n_agents, 40)
    """

    metadata = {"render_modes": ["human"]}  # no real-time render; we log CSV

    # --- world config ---
    SPACE_SIZE = 15.0          # L=W=H=15 m (reduced for faster learning)
    DT = 0.5                   # s
    V_MOVE = 1.0               # m/s command speed
    VIEW_RADIUS = 5.0          # m (compact search radius)
    MAX_STEPS = 400

    # --- wind ---
    W_STEADY = np.array([0.2, 0.1, 0.0])   # small steady wind (m/s)
    P_GUST = 0.005                         # gust trigger probability per step
    GUST_STRENGTH = (2.0, 4.0)             # m/s
    GUST_DURATION = (5, 15)                # steps
    GUST_YAW_DELTA = np.deg2rad(30.0)      # rad, yaw perturbation

    # --- drone ---
    AGENT_RADIUS = 0.3
    NUM_OBSTACLES = 5          # more obstacles for complex search scenarios
    OBST_SIZE_RANGE = (2.0, 4.0)

    def __init__(self, n_agents=3, render_mode=None, seed=None):
        super().__init__()
        self.n_agents = n_agents
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed)

        # action: flattened (n_agents, 3) continuous direction vectors
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_agents * 3,), dtype=np.float32
        )
        # obs: n_agents * 40 dims
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_agents * 40,), dtype=np.float32
        )

        # internal state
        self.agent_pos: np.ndarray = None          # (N, 3)
        self.agent_yaw: np.ndarray = None          # (N,)
        self.agent_yaw_dot: np.ndarray = None      # (N,)
        self.agent_memory: np.ndarray = None       # (N, 8)  octant memory
        self.target_pos: np.ndarray = None         # (3,)
        self.obstacles: List[Dict] = []            # list of {center, size, min, max}
        self.gust_active: np.ndarray = None        # (N,) bool
        self.gust_timer: np.ndarray = None         # (N,) int
        self.gust_vector: np.ndarray = None        # (N, 3)
        self.visited: set = set()                  # visited grid coords (int)
        self.step_count = 0
        self.trajectory: List[List] = []           # [step, aid, x, y, z, found]

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.step_count = 0
        self.trajectory = []
        self.visited.clear()

        self._generate_obstacles()
        self._generate_target()
        self._generate_agents()

        self.gust_active = np.zeros(self.n_agents, dtype=bool)
        self.gust_timer = np.zeros(self.n_agents, dtype=int)
        self.gust_vector = np.zeros((self.n_agents, 3), dtype=np.float32)
        self._prev_pos = self.agent_pos.copy()

        obs = self._get_obs()
        info = {"target": self.target_pos.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(self.n_agents, 3)

        # --- 1. wind gust trigger / countdown ---
        for i in range(self.n_agents):
            if self.gust_active[i]:
                self.gust_timer[i] -= 1
                if self.gust_timer[i] <= 0:
                    self.gust_active[i] = False
                    self.gust_vector[i] = 0.0
            else:
                if self.np_random.random() < self.P_GUST:
                    self.gust_active[i] = True
                    self.gust_timer[i] = int(self.np_random.integers(*self.GUST_DURATION))
                    strength = self.np_random.uniform(*self.GUST_STRENGTH)
                    dir_vec = self.np_random.normal(size=3)
                    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
                    self.gust_vector[i] = dir_vec * strength

        # --- 2. record prev signals for shaping reward ---
        prev_signals = np.array([
            np.exp(-np.linalg.norm(self.agent_pos[i] - self.target_pos) / 8.0)
            for i in range(self.n_agents)
        ], dtype=np.float32)
        self._prev_pos = self.agent_pos.copy()

        # --- 3. move each agent ---
        for i in range(self.n_agents):
            act_vec = action[i]
            act_norm = np.linalg.norm(act_vec)
            if act_norm > 0.01:
                v_cmd = (act_vec / act_norm) * self.V_MOVE
            else:
                v_cmd = np.zeros(3, dtype=np.float32)
            if self.gust_active[i]:
                v_cmd *= 0.5                     # 50% authority during gust

            # position update with wind
            dp = v_cmd * self.DT + self.W_STEADY * self.DT
            if self.gust_active[i]:
                dp += self.gust_vector[i] * self.DT
            self.agent_pos[i] += dp

            # yaw update
            old_yaw = self.agent_yaw[i]
            if np.linalg.norm(v_cmd[:2]) > 0.01:
                desired_yaw = np.arctan2(v_cmd[1], v_cmd[0])
                # smooth yaw tracking
                dyaw = desired_yaw - old_yaw
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                self.agent_yaw[i] += np.clip(dyaw, -0.2, 0.2)
            if self.gust_active[i]:
                self.agent_yaw[i] += self.np_random.uniform(
                    -self.GUST_YAW_DELTA, self.GUST_YAW_DELTA
                )
            self.agent_yaw[i] = (self.agent_yaw[i] + 2 * np.pi) % (2 * np.pi)
            self.agent_yaw_dot[i] = (self.agent_yaw[i] - old_yaw) / self.DT

            # boundary clamp (soft)
            self.agent_pos[i] = np.clip(self.agent_pos[i], 0.0, self.SPACE_SIZE)

        # --- 4. collisions, target, rewards ---
        reward, terminated = self._compute_reward_and_done(action, prev_signals)
        truncated = self.step_count >= self.MAX_STEPS

        # --- 4. logging ---
        found_flag = 1 if terminated else 0
        for i in range(self.n_agents):
            self.trajectory.append([
                self.step_count, i,
                self.agent_pos[i, 0], self.agent_pos[i, 1], self.agent_pos[i, 2],
                found_flag
            ])

        obs = self._get_obs()
        info = {"step": self.step_count}
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def _generate_obstacles(self):
        """Generate obstacles with uniform spatial distribution (partition-based)."""
        self.obstacles = []
        # divide space into quadrants, place one obstacle per quadrant for uniformity
        quadrants = [
            (0, self.SPACE_SIZE/2, 0, self.SPACE_SIZE/2, 0, self.SPACE_SIZE/2),           # lower-left-front
            (self.SPACE_SIZE/2, self.SPACE_SIZE, 0, self.SPACE_SIZE/2, 0, self.SPACE_SIZE/2),  # lower-right-front
            (0, self.SPACE_SIZE/2, self.SPACE_SIZE/2, self.SPACE_SIZE, 0, self.SPACE_SIZE/2),  # upper-left-front
            (self.SPACE_SIZE/2, self.SPACE_SIZE, self.SPACE_SIZE/2, self.SPACE_SIZE, 0, self.SPACE_SIZE/2), # upper-right-front
            (self.SPACE_SIZE/4, 3*self.SPACE_SIZE/4, self.SPACE_SIZE/4, 3*self.SPACE_SIZE/4,
             self.SPACE_SIZE/2, self.SPACE_SIZE),  # center-back
        ]
        chosen = self.np_random.choice(len(quadrants), size=self.NUM_OBSTACLES, replace=False)
        for qid in chosen:
            x0, x1, y0, y1, z0, z1 = quadrants[qid]
            size = self.np_random.uniform(1.5, 3.0, size=3)
            center = np.array([
                self.np_random.uniform(x0 + 1.0, x1 - 1.0),
                self.np_random.uniform(y0 + 1.0, y1 - 1.0),
                self.np_random.uniform(z0 + 1.0, z1 - 1.0),
            ], dtype=np.float32)
            half = size / 2.0
            self.obstacles.append({
                "center": center,
                "size": size,
                "min": center - half,
                "max": center + half,
            })

    def _generate_target(self):
        while True:
            pos = self.np_random.uniform(2.0, self.SPACE_SIZE - 2.0, size=3)
            if not self._inside_any_obstacle(pos):
                self.target_pos = pos
                break

    def _generate_agents(self):
        # spawn on 6 faces, far from target (>10 m)
        faces = [
            np.array([0.0, 0.5, 0.5]),
            np.array([1.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 1.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.5, 1.0]),
        ]
        chosen = self.np_random.choice(len(faces), size=self.n_agents, replace=False)
        self.agent_pos = np.zeros((self.n_agents, 3), dtype=np.float32)
        for i, fid in enumerate(chosen):
            self.agent_pos[i] = faces[fid] * self.SPACE_SIZE
            # add small jitter
            self.agent_pos[i] += self.np_random.normal(0, 0.5, size=3)
            self.agent_pos[i] = np.clip(self.agent_pos[i], 0.0, self.SPACE_SIZE)

        self.agent_yaw = np.zeros(self.n_agents, dtype=np.float32)
        self.agent_yaw_dot = np.zeros(self.n_agents, dtype=np.float32)
        # yaw roughly toward center
        center = np.array([self.SPACE_SIZE / 2] * 3)
        for i in range(self.n_agents):
            vec = center - self.agent_pos[i]
            self.agent_yaw[i] = np.arctan2(vec[1], vec[0])
        self.agent_memory = np.zeros((self.n_agents, 8), dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward & termination
    # ------------------------------------------------------------------
    def _compute_reward_and_done(self, action, prev_signals):
        reward = 0.0
        terminated = False

        # --- time penalty ---
        reward -= 0.02

        for i in range(self.n_agents):
            curr_dist = np.linalg.norm(self.agent_pos[i] - self.target_pos)

            # --- target found ? ---
            if curr_dist < 3.0:
                terminated = True
                reward += 500.0
                return reward, terminated

            # --- signal shaping reward (blind search: only signal strength, no direction) ---
            curr_signal = np.exp(-curr_dist / 8.0)
            signal_delta = curr_signal - prev_signals[i]  # positive if closer
            reward += 20.0 * signal_delta

            # --- obstacle / boundary collision ---
            if self._inside_any_obstacle(self.agent_pos[i]):
                reward -= 0.05
            if np.any(self.agent_pos[i] <= 0.01) or np.any(self.agent_pos[i] >= self.SPACE_SIZE - 0.01):
                reward -= 0.05

            # --- teammate too close ---
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.agent_pos[i] - self.agent_pos[j]) < 0.6:
                    reward -= 0.5

            # --- exploration reward (increased to encourage coverage) ---
            grid = tuple((self.agent_pos[i] / 1.0).astype(int))
            if grid not in self.visited:
                self.visited.add(grid)
                reward += 1.0
            else:
                reward -= 0.01

            # --- attitude stability reward (wind robustness) ---
            reward -= 0.1 * abs(self.agent_yaw_dot[i])
            cmd_yaw = self.agent_yaw[i]
            yaw_err = 0.0
            reward -= 0.05 * abs(yaw_err)

            # --- energy penalty (fighting wind) ---
            act_vec = action[i]
            act_norm = np.linalg.norm(act_vec)
            if act_norm > 0.01:
                v_cmd = (act_vec / act_norm) * self.V_MOVE
            else:
                v_cmd = np.zeros(3, dtype=np.float32)
            w_meas = self.W_STEADY + (self.gust_vector[i] if self.gust_active[i] else 0.0)
            reward -= 0.02 * np.linalg.norm(v_cmd - w_meas)

        return reward, terminated

    # ------------------------------------------------------------------
    # Observation (40 dims per agent)
    # ------------------------------------------------------------------
    def _get_obs(self):
        obs_list = []
        for i in range(self.n_agents):
            p = self.agent_pos[i]
            yaw = self.agent_yaw[i]
            yaw_dot = self.agent_yaw_dot[i]

            # measured wind
            w_meas = self.W_STEADY.copy()
            if self.gust_active[i]:
                w_meas += self.gust_vector[i]

            # 1. self position (normalized)
            obs_p = p / self.SPACE_SIZE * 2.0 - 1.0

            # 2. actual velocity
            if hasattr(self, '_prev_pos'):
                vel = (p - self._prev_pos[i]) / self.DT
                obs_v = np.clip(vel / self.V_MOVE, -1.0, 1.0)
            else:
                obs_v = np.zeros(3, dtype=np.float32)

            # 3. yaw & yaw_dot
            obs_yaw = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)
            obs_yawdot = np.clip(yaw_dot / 2.0, -1.0, 1.0)

            # 4. measured wind
            obs_w = np.clip(w_meas / 5.0, -1.0, 1.0)

            # 5. nearest 3 obstacles relative positions
            obs_obs = np.zeros(9, dtype=np.float32)
            dists = []
            for obs in self.obstacles:
                d_vec = obs["center"] - p
                d = np.linalg.norm(d_vec)
                if d <= self.VIEW_RADIUS:
                    dists.append((d, d_vec))
            dists.sort(key=lambda x: x[0])
            for k in range(min(3, len(dists))):
                obs_obs[k * 3:(k + 1) * 3] = np.clip(dists[k][1] / self.VIEW_RADIUS, -1.0, 1.0)

            # 6. nearest (n_agents-1) teammates relative positions (up to 2)
            max_teammates = min(2, self.n_agents - 1)
            obs_team = np.zeros(6, dtype=np.float32)  # keep 6 dims for consistency
            tlist = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                d_vec = self.agent_pos[j] - p
                d = np.linalg.norm(d_vec)
                if d <= self.VIEW_RADIUS:
                    tlist.append((d, d_vec))
            tlist.sort(key=lambda x: x[0])
            for k in range(min(max_teammates, len(tlist))):
                obs_team[k * 3:(k + 1) * 3] = np.clip(tlist[k][1] / self.VIEW_RADIUS, -1.0, 1.0)

            # 7. target relative pos & visibility
            t_vec = self.target_pos - p
            t_dist = np.linalg.norm(t_vec)
            visible = float(t_dist <= self.VIEW_RADIUS)
            if visible:
                obs_t = np.clip(t_vec / self.VIEW_RADIUS, -1.0, 1.0)
            else:
                obs_t = np.zeros(3, dtype=np.float32)

            # 8. target signal strength (blind search: scalar only, no direction)
            t_dist = np.linalg.norm(self.target_pos - p)
            obs_signal = np.array([np.exp(-t_dist / 8.0)], dtype=np.float32)

            # 9. octant memory (8 dims)
            obs_mem = self.agent_memory[i].copy()

            agent_obs = np.concatenate([
                obs_p,       # 3
                obs_v,       # 3
                obs_yaw,     # 2
                [obs_yawdot],# 1
                obs_w,       # 3
                obs_obs,     # 9
                obs_team,    # 6
                obs_t,       # 3
                [visible],   # 1
                obs_signal,  # 1  (signal strength, replaces direction prior)
                obs_mem,     # 8
            ]).astype(np.float32)
            obs_list.append(agent_obs)

        return np.concatenate(obs_list)

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _inside_any_obstacle(self, pos: np.ndarray) -> bool:
        for obs in self.obstacles:
            if np.all(pos >= obs["min"] - self.AGENT_RADIUS) and np.all(pos <= obs["max"] + self.AGENT_RADIUS):
                return True
        return False

    def save_trajectory(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "agent_id", "x", "y", "z", "found_target"])
            writer.writerows(self.trajectory)

    def get_state_dict(self) -> Dict:
        """For visualization: return world state."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "agent_yaw": self.agent_yaw.copy(),
            "target_pos": self.target_pos.copy(),
            "obstacles": self.obstacles,
            "space_size": self.SPACE_SIZE,
        }
