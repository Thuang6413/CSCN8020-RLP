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

        # --- Observation space (unchanged) ---
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
        vector_to_target = self.target_pos - self.data.qpos[:3]
        return np.concatenate([
            vector_to_target,
            self.data.qvel[:3],
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

        # --- Added: Calculate out-of-bounds distance ---
        # Calculate radial (XY plane) out-of-bounds distance
        radial_distance = np.sqrt(self.data.qpos[0]**2 + self.data.qpos[1]**2)
        distance_outside_radial = max(0, radial_distance - self.vessel_radius)

        # Calculate axial (Z axis) out-of-bounds distance
        z_min, z_max = self.entrance_pos[2], self.exit_pos[2]
        distance_outside_z = 0
        if self.data.qpos[2] < z_min:
            distance_outside_z = z_min - self.data.qpos[2]
        elif self.data.qpos[2] > z_max:
            distance_outside_z = self.data.qpos[2] - z_max
        # --- End added ---

        mj_step(self.model, self.data)

        obs = self._get_obs()
        current_dist = np.linalg.norm(self.target_pos - self.data.qpos[:3])

        # --- Stage 2 reward logic ---

        # 1. Goal achievement reward (unchanged)
        goal_reward = 0
        terminated = bool(current_dist < 0.05)
        if terminated:
            goal_reward = 1000

        # 2. Nonlinear distance reward (unchanged)
        distance_reward = 10 * (1 - np.tanh(10 * current_dist))

        # 3. Time penalty (unchanged)
        time_penalty = -0.05

        # 4. Added! Out-of-bounds penalty
        # As long as there is any out-of-bounds, give a fixed, large penalty
        outside_penalty = 0
        if distance_outside_radial > 0 or distance_outside_z > 0:
            outside_penalty = -500  # Severe penalty
            terminated = True  # Out of bounds, episode ends immediately

        # Total reward
        reward = goal_reward + distance_reward + time_penalty + outside_penalty
        # --- End modification ---

        if self.step_count >= self.max_steps:
            terminated = True

        truncated = False
        info = {"clot_dist": current_dist, "flow_vel": self.flow_vel,
                "outside_penalty": outside_penalty}

        result = (obs, reward, terminated, truncated, info)
        return result

    def render(self):
        if self.render_mode not in ["human", "rgb_array"] or self.renderers is None:
            return None

        images = []
        for cam in self.camera_names:
            self.renderers[cam].update_scene(self.data, camera=cam)
            img = self.renderers[cam].render()
            img_bgr = np.ascontiguousarray(img[..., ::-1])

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
