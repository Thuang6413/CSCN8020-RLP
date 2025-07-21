import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco import mj_step
import cv2  # ✅ 新增
from mujoco import Renderer  # ✅ 新增


class BloodVesselEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            "../assets/blood_vessel_scene.xml")
        self.data = mujoco.MjData(self.model)

        # ✅ 初始化 Renderer（for render with OpenCV）
        self.renderer = Renderer(self.model, width=640, height=480)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.array([0.0, 0.0, 0.5])  # 重設位置
        self.data.qvel[:] = 0
        obs = np.concatenate(
            [self.data.qpos.copy(), self.data.qvel.copy()]).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mj_step(self.model, self.data)

        obs = np.concatenate(
            [self.data.qpos.copy(), self.data.qvel.copy()]).astype(np.float32)
        reward = self.data.qpos[0]  # 向 x 方向推進給獎勵
        done = bool(np.linalg.norm(self.data.qpos) > 2.0)
        info = {}
        return obs, reward, done, False, info

    def render(self):
        # ✅ 使用 renderer 取得影像，再用 OpenCV 顯示
        self.renderer.update_scene(self.data)
        img = self.renderer.render()
        img_bgr = img[..., ::-1]  # RGB 轉 BGR 給 OpenCV 用
        cv2.imshow("Blood Vessel Navigation", img_bgr)
        cv2.waitKey(1)

    def close(self):
        # ✅ 關閉 OpenCV 視窗
        cv2.destroyAllWindows()
