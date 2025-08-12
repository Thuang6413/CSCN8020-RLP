import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step
import cv2
from mujoco import Renderer
import os


class BloodVesselEnv(gym.Env):
    def __init__(self, space_size=1.2, vessel_radius=0.1, viscosity=15e-3, friction_coeff=1.0, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        xml_path = "../assets/blood_vessel_merge.xml"
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- Get IDs ---
        entrance_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_entrance")
        exit_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "vessel_exit")
        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.model.body_mocapid[target_id]
        self.entrance_pos = self.model.body_pos[entrance_id]
        self.exit_pos = self.model.body_pos[exit_id]

        # --- Environment parameters ---
        self.space_size = space_size
        self.vessel_radius = vessel_radius
        self.viscosity = viscosity
        self.friction_coeff = friction_coeff
        self.flow_rate = 2.29e-6
        self.flow_vel = self.flow_rate / (self.vessel_radius ** 2)
        self.max_magnetic_field = 10.0
        self.max_steps = 1000

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.camera_names = ["top_view", "side_view", "front_view"]
            self.renderers = {cam: Renderer(
                self.model, width=640, height=480) for cam in self.camera_names}
        else:
            self.renderers = None

        # --- Action space (unchanged) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # --- Observation space (major update) ---
        # New observation space:
        # 3 (vector to target) + 3 (self velocity) + 1 (flow velocity) + 1 (viscosity) = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.target_pos = np.zeros(3)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.entrance_pos.copy()
        self.data.qvel[:] = 0.0

        # Randomize target position (unchanged)
        self.target_pos = np.random.uniform(
            low=-self.vessel_radius, high=self.vessel_radius, size=3)
        self.target_pos[2] = np.random.uniform(
            low=self.entrance_pos[2], high=self.exit_pos[2])
        while np.sqrt(self.target_pos[0]**2 + self.target_pos[1]**2) > self.vessel_radius:
            self.target_pos[:2] = np.random.uniform(
                low=-self.vessel_radius, high=self.vessel_radius, size=2)

        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # --- Observation logic (major update) ---
        # Calculate vector from agent to target
        vector_to_target = self.target_pos - self.data.qpos[:3]

        # Combine new observation vector
        return np.concatenate([
            vector_to_target,          # 3D: direction and distance to target
            self.data.qvel[:3],        # 3D: self velocity
            # 2D: environment parameters
            np.array([self.flow_vel, self.viscosity])
        ]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        # ... (Action mapping and physics simulation unchanged) ...
        B_magnitude = (action[0] + 1) * self.max_magnetic_field / 2
        B_dir = action[1:4] / np.linalg.norm(
            action[1:4]) if np.linalg.norm(action[1:4]) > 0 else np.zeros(3)
        force = B_magnitude * B_dir * 1e-3
        self.data.ctrl[:3] = force
        flow_force = -self.flow_vel * self.viscosity * self.friction_coeff
        self.data.qfrc_applied[:3] = [flow_force, 0, 0]

        prev_dist = np.linalg.norm(self.target_pos - self.data.qpos[:3])
        mj_step(self.model, self.data)
        obs = self._get_obs()
        current_dist = np.linalg.norm(self.target_pos - self.data.qpos[:3])

        # --- Use the previously validated "aggressive reward" logic ---
        goal_reward = 0
        terminated = bool(current_dist < 0.05)
        if terminated:
            goal_reward = 1000

        progress = prev_dist - current_dist
        if progress > 0:
            distance_reward = progress * 200
        else:
            distance_reward = -2.0

        time_penalty = -1.0
        reward = goal_reward + distance_reward + time_penalty

        if self.step_count >= self.max_steps:
            terminated = True

        truncated = False
        info = {"clot_dist": current_dist,
                "flow_vel": self.flow_vel, "outside_penalty": 0}
        return obs, reward, terminated, truncated, info

    def render(self):
        # ... (Render function unchanged, but we can update displayed text) ...
        if self.render_mode not in ["human", "rgb_array"] or self.renderers is None:
            return None

        images = []
        for cam in self.camera_names:
            self.renderers[cam].update_scene(self.data, camera=cam)
            img = self.renderers[cam].render()
            img_bgr = np.ascontiguousarray(img[..., ::-1])

            # Update displayed text
            clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
            cv2.putText(img_bgr, f"Dist: {clot_dist:.4f} m ({cam})", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_bgr, f"Flow: {self.flow_vel*1000:.2f} mm/s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            images.append(img_bgr)

        combined_img = np.hstack(images)
        if self.render_mode == "human":
            cv2.imshow("Blood Vessel Navigation", combined_img)
            cv2.waitKey(1)
        return combined_img

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()
