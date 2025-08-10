import os
import gymnasium as gym
import multiprocessing
import cv2
import numpy as np
import time
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
    # Create environment
    try:
        num_envs = 1  # Start with 1 to debug
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
        print(f"Environment: {env}")
        # Check only the first environment
        check_env(make_env(0)())
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit(1)

    # Load existing model (optional, for continued training)
    model_path = "sac_blood_vessel.zip"
    if os.path.exists(model_path):
        try:
            model = SAC.load(model_path, env=env)
            print(f"Loaded existing model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-3,
                buffer_size=1000000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                use_sde=False,
                tensorboard_log="./tensorboard_logs/"  # Enable TensorBoard logging
            )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=1000000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            use_sde=False,
            tensorboard_log="./tensorboard_logs/"  # Enable TensorBoard logging
        )

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, seed=100)])

    # Define evaluation callback with dynamic best_model_save_path
    # Use timestamp as run_id to ensure uniqueness
    run_id = f"SAC_{int(time.time())}"  # Unique run ID based on timestamp
    best_model_save_path = os.path.join("./best_model/", run_id)
    # Create directory if it doesn't exist
    os.makedirs(best_model_save_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path="./tensorboard_logs/",
        eval_freq=10000,  # Evaluate every 10,000 timesteps
        n_eval_episodes=10,  # Run 10 episodes per evaluation
        deterministic=True,  # Use deterministic policy for evaluation
        render=False
    )

    # Training loop with rendering for main environment
    total_timesteps = 100000  # Increase for more training
    render_frequency = 100  # Render every 100 steps
    timestep = 0
    obs = env.reset()

    try:
        # Train model with callback
        model.learn(total_timesteps=total_timesteps,
                    callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        # Save model and close environment
        model.save(f"sac_blood_vessel_new_{run_id}")
        env.close()
        eval_env.close()
        cv2.destroyAllWindows()
