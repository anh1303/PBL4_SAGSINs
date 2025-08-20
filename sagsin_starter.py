#!/usr/bin/env python3
"""
SAGSIN Starter (SimPy + Gymnasium + Stable-Baselines3)
------------------------------------------------------
A minimal, self-contained starter that:

1) Creates a simplified SAGSIN network simulation using SimPy
2) Wraps it as a Gymnasium environment
3) Trains a PPO agent from Stable-Baselines3 to minimize latency

Tested with:
- Python 3.9+
- pip install simpy gymnasium stable-baselines3

Quickstart:
-----------
python sagsin_starter.py --train_steps 2000
python sagsin_starter.py --eval_episodes 5

Notes:
------
- This is an abstracted model for fast prototyping AI logic.
- You can extend the topology, link models, and state features as needed.
"""

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO

import numpy as np
import simpy
import gymnasium as gym
from gymnasium import spaces

# Optional: only import SB3 if actually training (so users can run env-only without SB3)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


# -----------------------------
# Network Model (SimPy)
# -----------------------------

@dataclass
class LinkSpec:
    """Specification of a link between two nodes."""
    a: str
    b: str
    bandwidth_mbps: float       # Mbps
    prop_delay_ms: float        # ms
    error_rate: float = 0.0     # drop probability (0..1)

    def key(self) -> Tuple[str, str]:
        return tuple(sorted((self.a, self.b)))


class Packet:
    """A tiny packet model for latency measurement."""
    __slots__ = ("size_kb", "ts_created", "flow_id")
    def __init__(self, size_kb: float, ts_created: float, flow_id: int):
        self.size_kb = size_kb
        self.ts_created = ts_created
        self.flow_id = flow_id


class Link:
    """
    SimPy process representing a duplex link with simple serialization delay and propagation delay.
    We model the link as a single server with service time = transmission + propagation.
    """
    def __init__(self, env: simpy.Environment, spec: LinkSpec):
        self.env = env
        self.spec = spec
        self.name = f"{spec.a}<->{spec.b}"
        # One packet at a time (simple FIFO)
        self.res = simpy.Resource(env, capacity=1)
        self.bytes_in_transit = 0.0   # for state features (approximate utilization)

    @property
    def capacity_kBps(self) -> float:
        # Mbps -> kB/s (1 Mb = 10^6 bits, 1 byte = 8 bits). Approximate: 1 Mbps ≈ 125 kB/s
        return self.spec.bandwidth_mbps * 125.0

    def other_end(self, node: str) -> str:
        return self.spec.b if node == self.spec.a else self.spec.a

    def transmission_time(self, pkt_kB: float) -> float:
        # seconds
        return pkt_kB / max(self.capacity_kBps, 1e-6)

    def propagation_time(self) -> float:
        # seconds
        return self.spec.prop_delay_ms / 1000.0

    def busy_fraction(self) -> float:
        # proxy for utilization; not strictly accurate
        return min(self.bytes_in_transit / max(self.capacity_kBps, 1e-6), 1.0)

    def __repr__(self) -> str:
        return f"Link({self.name}, {self.spec.bandwidth_mbps}Mbps/{self.spec.prop_delay_ms}ms)"


class SAGSINSim:
    """
    Minimal SAGSIN simulator: nodes connected via links.
    One main flow: SHIP -> GROUND2 with a relay decision each step (UAV1, UAV2, or SAT).
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(int(seed) if seed is not None else None)
        self.env = simpy.Environment()
        self.now = lambda: self.env.now  # seconds

        # Nodes
        self.nodes = ["SAT", "UAV1", "UAV2", "GROUND1", "GROUND2", "SHIP"]

        # Link specs (you can expand/modify these)
        self.link_specs: List[LinkSpec] = [
            LinkSpec("SHIP", "UAV1", bandwidth_mbps=30, prop_delay_ms=15),
            LinkSpec("SHIP", "UAV2", bandwidth_mbps=25, prop_delay_ms=18),
            LinkSpec("SHIP", "SAT",  bandwidth_mbps=10, prop_delay_ms=40),
            LinkSpec("UAV1", "GROUND1", bandwidth_mbps=50, prop_delay_ms=10),
            LinkSpec("UAV2", "GROUND1", bandwidth_mbps=45, prop_delay_ms=10),
            LinkSpec("SAT",  "GROUND1", bandwidth_mbps=20, prop_delay_ms=30),
            LinkSpec("GROUND1", "GROUND2", bandwidth_mbps=80, prop_delay_ms=5),
        ]
        # Build link objects
        self.links: Dict[Tuple[str, str], Link] = {}
        for spec in self.link_specs:
            self.links[spec.key()] = Link(self.env, spec)

        # Metrics per-step
        self.step_pkts: List[Packet] = []
        self.step_latencies: List[float] = []
        self.step_drops: int = 0

        # Load / queue proxies per link (approximate)
        self.link_loads: Dict[Tuple[str, str], float] = {k: 0.0 for k in self.links.keys()}

    def get_link(self, u: str, v: str) -> Link:
        return self.links[tuple(sorted((u, v)))]

    def maybe_drop(self, link: Link) -> bool:
        return self.rng.random() < link.spec.error_rate

    def send_over_link(self, pkt: Packet, u: str, v: str):
        """SimPy process: send one packet from u to v over the link."""
        link = self.get_link(u, v)
        with link.res.request() as req:
            yield req

            # Transmission + propagation
            tx = link.transmission_time(pkt.size_kb)
            prop = link.propagation_time()

            link.bytes_in_transit += pkt.size_kb
            yield self.env.timeout(tx + prop)
            link.bytes_in_transit = max(link.bytes_in_transit - pkt.size_kb, 0.0)

            # Random drop
            if self.maybe_drop(link):
                self.step_drops += 1
                return False  # dropped
            return True  # delivered at v

    def path_send(self, pkt: Packet, path: List[str]):
        """Send a packet along a path of nodes (e.g., ['SHIP','UAV1','GROUND1','GROUND2'])."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            ok = yield self.env.process(self.send_over_link(pkt, u, v))
            if not ok:
                return False
        return True

    def generate_traffic_step(self, relay_choice: int, step_duration_s: float = 1.0):
        """
        One control step: choose relay (0=UAV1, 1=UAV2, 2=SAT).
        We generate a few packets for flow SHIP->GROUND2 and deliver via chosen relay path.
        """
        # Map action to path
        if relay_choice == 0:
            path = ["SHIP", "UAV1", "GROUND1", "GROUND2"]
        elif relay_choice == 1:
            path = ["SHIP", "UAV2", "GROUND1", "GROUND2"]
        else:
            path = ["SHIP", "SAT", "GROUND1", "GROUND2"]

        # Randomize traffic amount for this step
        num_packets = self.rng.randint(5, 12)
        for i in range(num_packets):
            size_kb = self.rng.uniform(8, 64)  # small to medium packets
            pkt = Packet(size_kb=size_kb, ts_created=self.now(), flow_id=0)
            self.step_pkts.append(pkt)

            def _deliver(pkt=pkt, path=path):
                start = self.now()
                ok = yield self.env.process(self.path_send(pkt, path))
                if ok:
                    lat = self.now() - start
                    self.step_latencies.append(lat)

            self.env.process(_deliver())

        # Let the system run for the duration of the control step
        yield self.env.timeout(step_duration_s)

        # Update link load proxies
        for key, link in self.links.items():
            self.link_loads[key] = 0.9 * self.link_loads[key] + 0.1 * link.busy_fraction()

    def reset_metrics(self):
        self.step_pkts.clear()
        self.step_latencies.clear()
        self.step_drops = 0


# -----------------------------
# Gymnasium Environment Wrapper
# -----------------------------

class SAGSINEnv(gym.Env):
    """
    Gymnasium wrapper around SAGSINSim.
    Observation:
        - 3 relay utilization proxies (SHIP-UAV1, SHIP-UAV2, SHIP-SAT)
        - GROUND1-GROUND2 load proxy
        - Last step latency stats: mean, p95 (clipped)
        - Last step drop count (clipped)
    Action:
        - Discrete(3): choose {UAV1, UAV2, SAT} as relay for the next step
    Reward:
        - Negative of p95 latency (sec), minus drop penalty
    Episode length:
        - Fixed number of steps (e.g., 100)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, episode_len: int = 100, seed: int = 42):
        super().__init__()
        self.episode_len = episode_len
        self._step = 0
        self.sim = SAGSINSim(seed=seed)

        # Observation space (continuous vector)
        # [load_ship_uav1, load_ship_uav2, load_ship_sat, load_g1_g2, mean_lat, p95_lat, drops]
        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(7,), dtype=np.float32
        )
        # Actions: 0=UAV1, 1=UAV2, 2=SAT
        self.action_space = spaces.Discrete(3)

        self.last_obs = None

    def _get_obs(self) -> np.ndarray:
        # Extract features
        ls = self.sim.link_loads
        load_ship_uav1 = ls[tuple(sorted(("SHIP", "UAV1")))]
        load_ship_uav2 = ls[tuple(sorted(("SHIP", "UAV2")))]
        load_ship_sat  = ls[tuple(sorted(("SHIP", "SAT")))]
        load_g1_g2     = ls[tuple(sorted(("GROUND1", "GROUND2")))]

        lats = self.sim.step_latencies
        mean_lat = float(np.mean(lats)) if lats else 0.5
        p95_lat  = float(np.percentile(lats, 95)) if lats else 0.7
        drops    = float(self.sim.step_drops)

        # Clip to keep observation bounded (avoid inf/NaN)
        mean_lat = float(np.clip(mean_lat, 0.0, 5.0))
        p95_lat  = float(np.clip(p95_lat, 0.0, 5.0))
        drops    = float(np.clip(drops, 0.0, 20.0))

        obs = np.array([
            load_ship_uav1, load_ship_uav2, load_ship_sat,
            load_g1_g2, mean_lat, p95_lat, drops
        ], dtype=np.float32)

        # Normalize lightly
        obs[:4] = np.clip(obs[:4], 0.0, 1.0)  # loads are 0..1 by design

        return obs

    def _compute_reward(self) -> float:
        lats = self.sim.step_latencies
        p95_lat = float(np.percentile(lats, 95)) if lats else 1.0
        drops = self.sim.step_drops
        # Reward: minimize latency and drops
        reward = -p95_lat - 0.2 * drops
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Recreate the simulator for a fresh episode
        self.sim = SAGSINSim(seed=self.np_random.integers(0, 10_000) if seed is None else seed)
        self.sim.reset_metrics()
        self._step = 0
        # Warm-up one step with default relay (UAV1) to populate state
        self.sim.env.process(self.sim.generate_traffic_step(relay_choice=0, step_duration_s=0.5))
        self.sim.env.run(until=self.sim.env.now + 0.5)
        obs = self._get_obs()
        self.last_obs = obs
        info = {"message": "reset_ok"}
        return obs, info

    def step(self, action):
        action = int(action)
        self.sim.reset_metrics()
        # Run one control step with the chosen relay
        self.sim.env.process(self.sim.generate_traffic_step(relay_choice=action, step_duration_s=0.5))
        self.sim.env.run(until=self.sim.env.now + 0.5)

        obs = self._get_obs()
        reward = self._compute_reward()
        self._step += 1
        terminated = False
        truncated = self._step >= self.episode_len
        info = {
            "mean_latency": float(np.mean(self.sim.step_latencies)) if self.sim.step_latencies else None,
            "p95_latency": float(np.percentile(self.sim.step_latencies, 95)) if self.sim.step_latencies else None,
            "drops": self.sim.step_drops
        }
        self.last_obs = obs
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.last_obs is None:
            print("No observation yet.")
            return
        print(f"Step={self._step} Obs={self.last_obs}")

    def close(self):
        pass


# -----------------------------
# Train / Evaluate Utilities
# -----------------------------

def train_ppo(total_steps: int = 5000, n_envs: int = 4, episode_len: int = 100, seed: int = 42):
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not installed. Run: pip install stable-baselines3")

    def make_env():
        return SAGSINEnv(episode_len=episode_len, seed=seed)

    # Vectorized environments for faster training
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed, tensorboard_log=None)
    model.learn(total_timesteps=total_steps)
    return model


def evaluate(model, episodes: int = 5):
    env = SAGSINEnv(episode_len=100, seed=123)
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            done = terminated or truncated
        rewards.append(ep_rew)
        print(f"[Eval] Episode {ep+1}: return={ep_rew:.3f}")
    print(f"[Eval] Mean return over {episodes} episodes: {np.mean(rewards):.3f}")
    env.close()


# -----------------------------
# Main CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_steps", type=int, default=0, help="Train PPO for N timesteps (0 = skip training).")
    parser.add_argument("--eval_episodes", type=int, default=0, help="Evaluate trained model for N episodes.")
    args = parser.parse_args()

    # Quick env sanity check if no training
    if args.train_steps <= 0 and args.eval_episodes <= 0:
        print("Sanity check: running a random policy for a few steps...")
        env = SAGSINEnv(episode_len=10, seed=0)
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"rand action={action} reward={reward:.3f} info={info}")
            done = terminated or truncated
        env.close()
        return

    model = None
    if args.train_steps > 0:
        print(f"Training PPO for {args.train_steps} timesteps...")
        model = train_ppo(total_steps=args.train_steps)

    if args.eval_episodes > 0:
        if model is None:
            if not SB3_AVAILABLE:
                raise RuntimeError("Stable-Baselines3 not installed to load a model for evaluation.")
            # If no training just create an untrained model for demo
            env = SAGSINEnv()
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=10000)
        print(f"Evaluating for {args.eval_episodes} episodes...")
        evaluate(model, episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
