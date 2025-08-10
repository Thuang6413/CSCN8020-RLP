from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from blood_vessel_env import BloodVesselEnv
import cv2
import numpy as np

# Define environment creation function for VecEnv


def make_env():
    def _init():
        env = BloodVesselEnv(render_mode="rgb_array")
        return env
    return _init


# Initialize environment with DummyVecEnv
env = DummyVecEnv([make_env()])

# Load the trained PPO model
# Update model_path to point to the PPO model (e.g., from train_ppo.py)
model_path = "/mnt/d/Project/CSCN8020-RLP/scripts/best_model/PPO_1754835155/best_model.zip"
# If the PPO model is saved differently, adjust the path accordingly
model = PPO.load(model_path, env=env)

# Reset environment and run for a few steps
obs = env.reset()  # Returns a single observation array for VecEnv
done = False
step_count = 0
max_steps = 1000

while not done and step_count < max_steps:
    action, _states = model.predict(
        obs, deterministic=True)  # Use deterministic policy
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        # Ensure terminated and truncated are arrays
        if not isinstance(terminated, np.ndarray):
            terminated = np.array([terminated])
        if not isinstance(truncated, np.ndarray):
            truncated = np.array([truncated])
    elif len(step_result) == 4:
        obs, reward, done, info = step_result
        # Assume done combines terminated and truncated
        terminated = np.array([done])
        # Set truncated to False for VecEnv compatibility
        truncated = np.array([False])
    else:
        raise ValueError(f"Unexpected step result length: {len(step_result)}")
    step_count += 1
    # Extract data for the first environment (index 0)
    print(f"Step {step_count}: Reward={reward[0]:.2f}, Clot Distance={info[0]['clot_dist']:.6f}, "
          f"Collision Penalty={info[0]['collision_penalty']:.2f}, Terminated={terminated[0]}, Truncated={truncated[0]}")

    # Render the first environment
    img = env.env_method("render", indices=[0])[0]
    if img is not None:
        cv2.imshow("Blood Vessel Navigation", img)
        cv2.waitKey(1)

    # Check done for the first environment
    done = terminated[0] or truncated[0]

# Close the environment
env.close()
cv2.destroyAllWindows()
