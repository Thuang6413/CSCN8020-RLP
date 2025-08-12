import os
import gymnasium as gym
import time
# Ensure the updated environment is imported
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
        # Ensure using the environment designed for stage 2 with boundary penalty
        env = BloodVesselEnv(render_mode="rgb_array" if rank == 0 else None)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    # --- Create environment (unchanged) ---
    try:
        num_envs = 1
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        print(f"Stage 2 environment created.")
        # Check if the environment meets the requirements of stable-baselines3
        check_env(make_env(0)())
    except Exception as e:
        print(f"Environment creation failed: {e}")
        exit(1)

    # --- Model loading and learning rate setting ---
    # Strategy: Load the best model from stage 1 and use a new learning rate for learning new rules
    # This learning rate is higher than fine-tuning, which helps learn new knowledge, but lower than training from scratch for stability
    learning_rate = 1e-4  # 0.0001

    # Important! Please replace this path with the actual path to best_model.zip generated after stage 1 training
    # For example: "./best_model/SAC_xxxxxxxxxx_final_correct_env/best_model.zip"
    model_path = "./best_model/SAC_1754880889_final_correct_env/best_model.zip"

    if os.path.exists(model_path):
        try:
            # Load our trained stage 1 model
            model = SAC.load(model_path, env=env, learning_rate=learning_rate)
            print(f"Successfully loaded stage 1 model from {model_path}.")
            print(
                f"Starting stage 2 training, learning rate set to: {learning_rate}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            exit(1)
    else:
        print(
            f"Model file not found at {model_path}, cannot start stage 2 training. Please check the path.")
        exit(1)
    # --- End of modification ---

    # --- Evaluation and saving settings ---
    eval_env = DummyVecEnv([make_env(0, seed=100)])
    # Create a new, unique run ID for stage 2 training
    run_id = f"SAC_{int(time.time())}_stage2"
    best_model_save_path = os.path.join("./best_model/", run_id)
    os.makedirs(best_model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path="./tensorboard_logs/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Set new training steps for stage 2
    total_timesteps = 250000

    try:
        # Start training
        model.learn(total_timesteps=total_timesteps,
                    callback=eval_callback,
                    progress_bar=True,
                    reset_num_timesteps=True)  # Important! Restart counting for the new stage of learning
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        # Save the final model
        model.save(f"sac_stage2_final_{run_id}")
        env.close()
        eval_env.close()
        cv2.destroyAllWindows()
