import blenderproc as bproc
import numpy as np
from scipy.spatial.transform import Rotation as R 

scene = bproc.loader.load_blend("C:/Users/Bende/Documents/blender_hangar/scenes/a320_in_hangar_2.blend")

# Set up camera resolution and intrinsics.
resolution = 4504
focal_length_mm = 12
sensor_width_mm = 12.3
sensor_height_mm = 12.3

bproc.camera.set_resolution(resolution, resolution)
bproc.camera.set_intrinsics_from_blender_params(
    lens=focal_length_mm, 
    lens_unit="MILLIMETERS",
)

# Direct sensor configuration
cam = bproc.camera.get_camera()
cam.data.sensor_width = sensor_width_mm
cam.data.sensor_height = sensor_height_mm  # Optional for square sensors

# Define the manual offset.
offset = np.array([-43.55, -13.8, 0.0])
base_camera_loc = np.array([-15.403574981493046, 18.814318536920652, 23.0])
# Combined final location with the offset as in your bpy code.
final_camera_loc = base_camera_loc + offset

# Define the rotation as Euler angles in radians
rotation_euler = (0, 0, 1.5708)

# Convert the Euler angles to a rotation matrix using mathutils.
rotation_matrix = R.from_euler("xyz", rotation_euler).as_matrix()

# Build a 4x4 camera-to-world transformation matrix from the location and rotation.
cam2world_matrix = bproc.math.build_transformation_mat(final_camera_loc, rotation_matrix)

bproc.camera.add_camera_pose(cam2world_matrix)

# Render the scene
data = bproc.renderer.render()

# Write the rendering into an hdf5 file
bproc.writer.write_hdf5("output/", data)