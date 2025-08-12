import os
import gymnasium as gym
import time
# Ensure the latest environment is loaded
from blood_vessel_env import BloodVesselEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC


def make_env(rank, seed=0):
    def _init():
        env = BloodVesselEnv(render_mode="rgb_array" if rank == 0 else None)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    # --- Create environment ---
    try:
        num_envs = 1
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        print(
            f"Environment created. Observation space shape: {env.observation_space.shape}")
        check_env(make_env(0)())
    except Exception as e:
        print(f"Environment creation failed: {e}")
        exit(1)

    # --- Brand new model and training ---
    # Strategy: Train a brand new model from scratch because the observation space has changed
    learning_rate = 3e-4  # Use a standard, stable learning rate

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=500000,  # Use a larger buffer to handle complexity
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",  # Automatic entropy adjustment for exploration
        tensorboard_log="./tensorboard_logs/"
    )
    print("A brand new SAC model has been created and is ready for training from scratch.")

    # --- Evaluation and saving settings ---
    eval_env = DummyVecEnv([make_env(0, seed=100)])
    run_id = f"SAC_{int(time.time())}_final_correct_env"
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

    # Set enough steps for brand new training
    total_timesteps = 300000

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=eval_callback,
                    progress_bar=True,
                    reset_num_timesteps=True)  # Start counting from zero
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        model.save(f"sac_final_correct_env_{run_id}")
        env.close()
        eval_env.close()
        cv2.destroyAllWindows()
