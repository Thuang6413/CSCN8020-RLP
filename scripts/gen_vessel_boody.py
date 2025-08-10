import trimesh
import numpy as np

# mesh = trimesh.load('blood_vessel.obj')
# # Move centroid to origin
# mesh.apply_translation(
#     [-mesh.centroid[0], -mesh.centroid[1], -mesh.centroid[2]])
# mesh.export('blood_vessel_centered.obj')

# Parameter settings
body_height = 0.05        # Blood vessel length
body_radius = 0.01        # Blood vessel radius
disc_thickness = 0.0005   # Entrance/exit disc thickness

# Create the main body of the blood vessel (long cylinder)
vessel_body = trimesh.creation.cylinder(
    radius=body_radius, height=body_height, sections=32)
vessel_body.visual.face_colors = [200, 50, 50, 255]  # Red

# Create the entrance region (thin cylinder, located at the bottom)
entrance_disc = trimesh.creation.cylinder(
    radius=body_radius, height=disc_thickness, sections=32)
entrance_disc.apply_translation([0, 0, -body_height / 2 - disc_thickness / 2])
entrance_disc.visual.face_colors = [50, 200, 50, 255]  # Green

# Create the exit region (thin cylinder, located at the top)
exit_disc = trimesh.creation.cylinder(
    radius=body_radius, height=disc_thickness, sections=32)
exit_disc.apply_translation([0, 0, body_height / 2 + disc_thickness / 2])
exit_disc.visual.face_colors = [50, 50, 200, 255]  # Blue

# Combine all geometries
combined = trimesh.util.concatenate([vessel_body, entrance_disc, exit_disc])

# Export as .obj (with material .mtl)
combined.export("../assets/source/vessel_with_entrance_exit.obj")
print("Exported vessel_with_entrance_exit.obj file")
