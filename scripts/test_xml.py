import mujoco
from mujoco import viewer

# Load the model
model = mujoco.MjModel.from_xml_path('../assets/blood_vessel_scene.xml')
data = mujoco.MjData(model)

# Use viewer to visualize the model
viewer.launch(model, data)
