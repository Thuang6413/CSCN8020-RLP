import cv2
import numpy as np
import time  # Import time module for pause
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
# Key: Make sure to import the final version of the environment definition
from blood_vessel_env import BloodVesselEnv


def make_env():
    """
    Helper function to create the environment.
    """
    def _init():
        # Use our final environment with dynamic parameters
        # render_mode="rgb_array" is correct because we want to control the display in the main loop
        env = BloodVesselEnv(render_mode="rgb_array")
        return env
    return _init


# Initialize environment
env = DummyVecEnv([make_env()])

# --- Key! Load your final trained model ---
# Please replace this path with the actual path to best_model.zip generated after stage 3 training
# For example: "./best_model/SAC_xxxxxxxxxx_stage3_robust/best_model.zip"
model_path = "./best_model/SAC_1754890706_stage3_robust/best_model.zip"
model = SAC.load(model_path, env=env)

# --- Main loop, will keep running until manually closed ---
try:
    # This infinite loop keeps the program running
    while True:
        # --- Setup for a single episode ---
        obs = env.reset()
        done = False
        step_count = 0
        max_steps = 1000  # Keep consistent with max_steps during training

        print("New episode started! Environment reset, target position, flow velocity, and viscosity updated.")

        # --- Loop for a single episode ---
        while not done and step_count < max_steps:
            # Use the model to predict the next action
            action, _states = model.predict(obs, deterministic=True)

            # --- Correction start ---
            # DummyVecEnv's step() returns 4 arrays: obs, rewards, dones, infos
            # We need to unpack them
            obs, rewards, dones, infos = env.step(action)

            # Since we only have one environment, take the first element from the arrays
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            # --- Correction end ---

            step_count += 1

            # Get the rendered image from the environment
            img = env.env_method("render", indices=[0])[0]
            if img is not None:
                # Show the image, and indicate in the title how to quit
                cv2.imshow("Blood Vessel Navigation - Press 'q' to quit", img)

            # Check keyboard input, if 'q' is pressed, trigger interrupt to exit loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            # Now the 'done' variable will be updated correctly, and this loop will exit normally at the end of an episode

        # After an episode ends
        if done:
            print(f"Episode ended at step {step_count}.")
            print("A new episode will start in 2 seconds...")
            # Pause for 2 seconds to allow you to see the state at the end of the episode
            time.sleep(2)

except KeyboardInterrupt:
    # This part runs when you press 'q' or Ctrl+C
    print("\nExit command detected, closing program...")
finally:
    # In any case, gracefully close the environment and all OpenCV windows
    env.close()
    cv2.destroyAllWindows()
    print("Program closed.")
