import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('../assets/Z-Anatomy-Layer-5.xml')
data = mujoco.MjData(model)
print("Model loaded successfully!")

# 啟動視圖器
mujoco.viewer.launch(model, data)
