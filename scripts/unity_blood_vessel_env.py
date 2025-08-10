import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step
import websocket
import json
import threading
import time


class BloodVesselEnv(gym.Env):
    def __init__(self, space_size=0.05, viscosity=15e-3, friction_coeff=1.0, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <worldbody>
                <camera name="overview" pos="0 0 0.1" xyaxes="1 0 0 0 1 0" fovy="45"/>
                <body name="agent" pos="0 0 0">
                    <geom type="capsule" size="0.001875 0.0045" density="1000" rgba="0 0 1 1"/>
                    <freejoint name="agent_free"/>
                </body>
                <body name="target" mocap="true">
                    <geom type="sphere" size="0.002" rgba="1 0 0 1"/>
                </body>
                <body name="trajectory_root">
                </body>
            </worldbody>
            <actuator>
                <motor name="force_x" joint="agent_free"/>
                <motor name="force_y" joint="agent_free"/>
                <motor name="force_z" joint="agent_free"/>
            </actuator>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.space_size = space_size
        self.viscosity = viscosity
        self.friction_coeff = friction_coeff
        self.flow_rate = 2.29e-6
        self.flow_vel = self.flow_rate / (self.space_size ** 2)
        self.max_speed = 19e-3
        self.max_magnetic_field = 10.0
        self.max_steps = 1000
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.target_pos = np.zeros(3)
        self.step_count = 0
        self.mocap_id = self.model.body("target").mocapid
        self.trajectory = []
        self.trajectory_root_id = self.model.body("trajectory_root").id
        self.prev_clot_dist = np.sqrt(3 * (self.space_size ** 2))
        self.prev_pos = np.zeros(3)
        self.ws = None
        self.ws_thread = None
        self.start_websocket()

    def start_websocket(self):
        def on_open(ws):
            print("WebSocket connected to Unity")

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket closed")
        # Retry connection
        max_retries = 5
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                self.ws = websocket.WebSocketApp("ws://localhost:8765/mujoco",
                                                 on_open=on_open,
                                                 on_error=on_error,
                                                 on_close=on_close)
                self.ws_thread = threading.Thread(
                    target=self.ws.run_forever)
                self.ws_thread.daemon = True
                self.ws_thread.start()
                time.sleep(1)  # Wait for connection
                if self.ws.sock and self.ws.sock.connected:
                    break
            except Exception as e:
                print(
                    f"WebSocket connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        if not (self.ws.sock and self.ws.sock.connected):
            print("Failed to connect to Unity WebSocket server after retries")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Randomize agent position within space_size
        agent_pos = np.random.uniform(
            low=-self.space_size / 2, high=self.space_size / 2, size=3)
        self.data.qpos[:3] = agent_pos
        self.data.qvel[:] = 0.0
        # Randomize target position within space_size, ensure not too close to agent
        while True:
            target_pos = np.random.uniform(
                low=-self.space_size / 2, high=self.space_size / 2, size=3)
            if np.linalg.norm(target_pos - agent_pos) > 0.005:
                break
        self.target_pos = target_pos
        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.step_count = 0
        self.trajectory = [self.data.qpos[:3].copy()]
        self.prev_clot_dist = np.linalg.norm(
            self.data.qpos[:3] - self.target_pos)
        self.prev_pos = self.data.qpos[:3].copy()
        self.send_data_to_unity()
        return self._get_obs(), {}

    def _get_obs(self):
        clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
        B_magnitude = self.data.ctrl[0] * \
            self.max_magnetic_field / 2 + self.max_magnetic_field / 2
        return np.concatenate([
            self.data.qpos[:3],
            self.data.qvel[:3],
            np.array([self.flow_vel, self.viscosity,
                      clot_dist, B_magnitude])
        ]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        B_magnitude = (action[0] + 1) * self.max_magnetic_field / 2
        B_dir = action[1:4] / np.linalg.norm(
            action[1:4]) if np.linalg.norm(action[1:4]) > 0 else np.zeros(3)
        force = B_magnitude * B_dir * 1e-3
        self.data.ctrl[:3] = force
        flow_force = -self.flow_vel * self.viscosity * self.friction_coeff
        self.data.qfrc_applied[:3] = [flow_force, 0, 0]
        collision_force = 0
        for i in range(3):
            if abs(self.data.qpos[i]) > self.space_size / 2:
                collision_force += self.friction_coeff * \
                    abs(
                        self.data.qpos[i] - np.sign(self.data.qpos[i]) * self.space_size / 2)
                self.data.qpos[i] = np.clip(
                    self.data.qpos[i], -self.space_size / 2, self.space_size / 2)
        mj_step(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self.target_pos
        self.trajectory.append(self.data.qpos[:3].copy())
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)
        obs = self._get_obs()
        clot_dist = np.linalg.norm(self.data.qpos[:3] - self.target_pos)
        current_pos = self.data.qpos[:3].copy()
        reward = 10 * (self.prev_clot_dist - clot_dist) / \
            0.001 if clot_dist < self.prev_clot_dist else 0
        if np.all(np.abs(current_pos - self.prev_pos) > 0):
            reward += 1
        if clot_dist < 0.00001:
            reward += 100
        self.prev_clot_dist = clot_dist
        self.prev_pos = current_pos
        terminated = bool(clot_dist < 0.001 or self.step_count >= self.max_steps or np.linalg.norm(
            self.data.qpos[:3]) > self.space_size)
        truncated = False
        info = {"clot_dist": clot_dist, "flow_vel": self.flow_vel}
        self.send_data_to_unity()
        return obs, reward, terminated, truncated, info

    def send_data_to_unity(self):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            data = {
                "agent_pos": self.data.qpos[:3].tolist(),
                "target_pos": self.target_pos.tolist(),
                "trajectory": [pos.tolist() for pos in self.trajectory],
                "clot_dist": float(np.linalg.norm(self.data.qpos[:3] - self.target_pos)),
                "flow_vel": float(self.flow_vel)
            }
            try:
                self.ws.send(json.dumps(data))
            except Exception as e:
                print(f"Error sending data to Unity: {e}")

    def close(self):
        if self.ws:
            self.ws.close()
