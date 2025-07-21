import mujoco
from mujoco import viewer

# 載入模型
model = mujoco.MjModel.from_xml_path('../assets/blood_vessel_scene.xml')
data = mujoco.MjData(model)

# 使用 viewer 查看模型
viewer.launch(model, data)
