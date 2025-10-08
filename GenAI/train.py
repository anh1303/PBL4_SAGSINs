import gymnasium as gym
import torch
import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from Environment import SagsEnv

# ======== Custom Logging Callback ==========
class RewardLoggingCallback(BaseCallback):
    def __init__(self, print_freq=100, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.total_eps = 0
        self.running_mean = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        info = infos[0]
        if info.get("score") is not None:
            self.total_eps += 1
            score = info["score"]
            self.running_mean += (score - self.running_mean) / self.total_eps

            if self.total_eps % self.print_freq == 0:
                print(f"[Episode {self.total_eps}] Mean Score (running): {self.running_mean:.3f}")
                #Display the most recent episode score for latency, reliability, uplink, downlink and time
                # if info.get("latency") is not None:
                #     print(f"  Latency: {info['latency']:.3f}, Reliability: {info['reliability']:.3f}, Uplink: {info['uplink']:.3f}, Downlink: {info['downlink']:.3f}, Timeout: {info['timeout']:.3f}")
        return True

def make_env():
    env = SagsEnv()
    env = Monitor(env)  # adds reward & episode length tracking
    return env

def main():
    # ======== Environment ==========
    env = make_env()

    # ======== Logger Setup ==========
    log_dir = "./logs/qrdqn_sagsenv/"
    new_logger = configure(log_dir, ["tensorboard", "csv"])


    # ======== Model Setup ==========
    model = QRDQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=512,
        tau=1.0,  # Update rate for target network (important for QRDQN)
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1000,  # How often to update target network (in steps)
        verbose=0,
        tensorboard_log=log_dir,
        device="auto",
        policy_kwargs=dict(
            net_arch=[512, 512],  # Larger network for high-dimensional observations
            n_quantiles=50  # Move n_quantiles here for QRDQN
        )
    )

    model.set_logger(new_logger)

    # ======== Training ==========
    callback = RewardLoggingCallback(print_freq=100)
    model.learn(total_timesteps=200_000, callback=callback, log_interval=100)

    # ======== Save Model ==========
    model.save("qrdqn_model")
    model.save_replay_buffer("qrdqn_buffer")

    print("✅ Training finished and model saved.")

if __name__ == "__main__":
    main()
