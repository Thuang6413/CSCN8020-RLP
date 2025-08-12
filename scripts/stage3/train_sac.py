import os
import gymnasium as gym
import time
# Make sure to import the latest environment above
from blood_vessel_env import BloodVesselEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from typing import Callable


def make_env(rank, seed=0):
    """
    Helper function to create the environment.
    """
    def _init():
        # Ensure using the environment designed for stage 3 with dynamic parameters
        env = BloodVesselEnv(render_mode="rgb_array" if rank == 0 else None)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    # --- Create environment ---
    try:
        num_envs = 1
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        print(f"Stage 3 (dynamic) environment created.")
        check_env(make_env(0)())
    except Exception as e:
        print(f"Environment creation failed: {e}")
        exit(1)

    # --- Model loading and learning rate setting ---
    # Strategy: Load the best model from stage 2 and fine-tune with a smaller learning rate to adapt to the dynamic environment
    learning_rate = 5e-5  # 0.00005, use a smaller learning rate for stable fine-tuning

    # Important! Please replace this path with the actual path to the best_model.zip generated after stage 2 training
    model_path = "./best_model/SAC_1754885992_stage2/best_model.zip"

    if os.path.exists(model_path):
        try:
            model = SAC.load(model_path, env=env, learning_rate=learning_rate)
            print(f"Successfully loaded stage 2 model from {model_path}.")
            print(
                f"Starting stage 3 fine-tuning, learning rate set to: {learning_rate}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            exit(1)
    else:
        print(
            f"Model file not found at {model_path}, cannot start stage 3 training. Please check the path.")
        exit(1)
    # --- End of modification ---

    # --- Evaluation and saving settings ---
    eval_env = DummyVecEnv([make_env(0, seed=100)])
    run_id = f"SAC_{int(time.time())}_stage3_robust"
    best_model_save_path = os.path.join("./best_model/", run_id)
    os.makedirs(best_model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path="./tensorboard_logs/",
        eval_freq=10000,
        n_eval_episodes=20,  # Increase evaluation episodes for more stable average
        deterministic=True,
        render=False
    )

    # Set new training steps for stage 3 fine-tuning
    total_timesteps = 300000

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=eval_callback,
                    progress_bar=True,
                    reset_num_timesteps=True)  # Restart counting for the new stage of learning
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        model.save(f"sac_stage3_robust_final_{run_id}")
        env.close()
        eval_env.close()
