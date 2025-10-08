import gym
import torch
import numpy as np
from sb3_contrib import SACDiscrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from Enviroment import SagsEnv

# ======== Custom Logging Callback ==========
class RewardLoggingCallback(BaseCallback):
    def __init__(self, print_freq=10, verbose=1):
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
            # Update running mean efficiently
            self.running_mean += (score - self.running_mean) / self.total_eps

            if self.total_eps % self.print_freq == 0:
                print(f"[Episode {self.total_eps}] Mean Score (running): {self.running_mean:.3f}")
        return True





def make_env():
    env = SagsEnv()
    env = Monitor(env)  # adds reward & episode length tracking
    return env


def main():
    # ======== Environment ==========
    env = make_env()

    # ======== Logger Setup ==========
    log_dir = "./logs/sac_discrete_sagsenv/"
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # ======== Model Setup ==========
    model = SACDiscrete(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Slightly lower for stability
        buffer_size=200000,   # Larger buffer
        learning_starts=5000, # More exploration before learning
        batch_size=512,       # Larger batch size if memory permits
        tau=0.01,            # Smoother target network updates
        gamma=0.99,
        train_freq=1,        # Train every step
        gradient_steps=1,    # Single gradient step per training
        ent_coef="auto",     # Let SAC tune entropy automatically
        target_update_interval=1,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )
    model.set_logger(new_logger)

    # ======== Training ==========
    callback = RewardLoggingCallback(check_freq=2000)
    model.learn(total_timesteps=200000, callback=callback, log_interval=10)

    # ======== Save Model ==========
    model.save("sac_discrete_model")
    model.save_replay_buffer("sac_discrete_buffer")
    
    # 2️⃣ Load the pretrained model
    # model = SACDiscrete.load("sac_discrete_model", env=env)

    # # 3️⃣ Optionally keep past experiences to stabilize further learning
    # if os.path.exists("sac_discrete_buffer.pkl"):
    #     model.load_replay_buffer("sac_discrete_buffer.pkl")
    #     print("Replay buffer loaded successfully (AI keeps old experience).")

    # # 4️⃣ Continue training — environment will reset itself naturally on episode end
    # model.learn(
    #     total_timesteps=200_000,
    #     callback=RewardLoggingCallback(print_freq=10),
    #     reset_num_timesteps=False  # continue global step count
    # )

    print("✅ Training finished and model saved.")


if __name__ == "__main__":
    main()
