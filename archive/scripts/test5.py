import blenderproc as bproc
import argparse
import numpy as np
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="examples/basics/semantic_segmentation/scene.blend", help="Path to the scene.blend file")
parser.add_argument('output_dir', nargs='?', default="1920_8gens", help="Path to where the final files will be saved")
parser.add_argument('--runs', type=int, default=1, help="Number of runs to perform")
args = parser.parse_args()

bproc.init()

# Load the objects into the scene
objs = bproc.loader.load_blend(args.scene)

# Define objects and their respective category IDs
object_categories = {
    1: [
        "air_conduit", "ceiling", "ceiling.001", "ceiling.002", "ceiling.003",
        "ceiling.004", "ceiling.005", "ceiling.006", "ceiling.007", "ceiling.008",
        "ceiling.009", "ceiling.010", "ceiling.011", "ceiling.012", "ceiling.013",
        "ceiling.lights", "ceiling.sprinkler", "ceiling_walkway", "ceiling_walkway.001",
        "ceiling_walkway.002", "details_offices", "door_large", "door_large.001",
        "door_large.002", "door_small", "door_small.001", "door_small.002",
        "electric_wires", "electric_wires.001", "floor", "floor.grids", "ground",
        "Hangar.walls", "main_doors", "offices_stairs", "pipes_back_wall",
        "pipes_side", "pipes_side.001", "pipes_side.002", "stairs.001", "stairs.002",
        "stairs.003", "stairs.004", "stairs.005", "stairs.006", "stairs.007",
        "stairs.008", "stairs.009", "stairs.010", "stairs.011", "stairs.012",
        "stairs.013", "walkway", "walkway.001", "walkway.002", "walkway.003",
        "walkway.004", "walkway.005", "walkway.006", "walkway.007", "walkway.009",
        "walkway.010", "walkway.011", "windows", "windows.001", "windows.002",
        "windows_offices"
    ],
    2: ["fuselage/wing"],
    3: ["nacelle"],
    4: ["rudder"],
    5: ["elevator"],
    6: [ "tyres", "Cylinder.018"],
    7: ["cockpit"],
    8: ["flaps"],
    9: [
        "forklift 2","forklift","forklift 3","forklift 4", "forklift 5", "forklift 6"
    ],
    10: [
        "scissor lift 3","scissor lift 2","scissor lift 4","scissor lift", "scissor lift 5", "scissor lift 6"
    ]
}

# Assign category IDs to the specified objects
for cat_id, obj_names in object_categories.items():
    for obj_name in obj_names:
        objs_with_name = bproc.filter.by_attr(objs, "name", obj_name)
        if objs_with_name:
            for obj in objs_with_name:
                obj.set_cp("category_id", cat_id)
        else:
            print(f"Warning: Object named '{obj_name}' not found in the scene.")

# Set unique category IDs for remaining objects
current_category_id = max(object_categories.keys()) + 1
for obj in objs:
    try:
        if obj.get_cp("category_id") is None:  # Only set if not already set
            obj.set_cp("category_id", current_category_id)
            current_category_id += 1
    except KeyError:
        obj.set_cp("category_id", current_category_id)
        current_category_id += 1

# Define the camera intrinsics and resolution from the provided K matrix

K_matrix = np.array([[3072.0000, 0.0000, 960.0000],
                     [0.0000, 3072.0000, 600.0000],
                     [0.0000, 0.0000, 1.0000]])

bproc.camera.set_intrinsics_from_K_matrix(K=K_matrix, image_width=1920, image_height=1200)


# Activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
# Enable segmentation masks (per class and per instance)
#bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])


# activate normal rendering
#bproc.renderer.enable_normals_output()
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# Perform multiple runs
for r in range(args.runs):
    # Clear all keyframes from the previous run
    bproc.utility.reset_keyframes()

    # Place the objects randomly for each run
    f1 = bproc.filter.one_by_attr(objs, "name", "forklift")
    f2 = bproc.filter.one_by_attr(objs, "name", "forklift 2")
    s1 = bproc.filter.one_by_attr(objs, "name", "scissor lift")
    s2 = bproc.filter.one_by_attr(objs, "name", "scissor lift 2")
    s3 = bproc.filter.one_by_attr(objs, "name", "scissor lift 3")
    f3 = bproc.filter.one_by_attr(objs, "name", "forklift 3")
    f4 = bproc.filter.one_by_attr(objs, "name", "forklift 4")
    f5 = bproc.filter.one_by_attr(objs, "name", "forklift 5")
    f6 = bproc.filter.one_by_attr(objs, "name", "forklift 6")
    s4 = bproc.filter.one_by_attr(objs, "name", "scissor lift 4")
    s5 = bproc.filter.one_by_attr(objs, "name", "scissor lift 5")
    s6 = bproc.filter.one_by_attr(objs, "name", "scissor lift 6")

    f1.set_location(np.random.uniform([23.2, 8.1, 0], [31.3, 8.1, 0]))
    f1.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    f2.set_location(np.random.uniform([-24.5, 15.4, 0], [-16.2, 7.9, 0]))
    f2.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s1.set_location(np.random.uniform([20.5, -.3, 1.5159], [28.704, -.3, 1.5159]))
    s1.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s2.set_location(np.random.uniform([-11, 23.4, 1.5159], [3.3, 18.05, 1.5159]))
    s2.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s3.set_location(np.random.uniform([2.7, 45, 1.5159], [7.3, 31, 1.5159]))
    s3.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    f3.set_location(np.random.uniform([28.1, 19, 0], [28.1, 24.5, 0]))
    f3.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    f4.set_location(np.random.uniform([39, 7.5, 0], [47.65, 7.5, 0]))
    f4.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    f5.set_location(np.random.uniform([13.1, -11.5, 0], [6.4, -19.5, 0]))
    f5.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    f6.set_location(np.random.uniform([.2, -20.5, 0], [.2, -11.3, 0]))
    f6.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s4.set_location(np.random.uniform([49, 16, 1.5159], [39, 20, 1.5159]))
    s4.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s5.set_location(np.random.uniform([8.7, -1.1, 1.5159], [5, 3.7, 1.5159]))
    s5.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))

    s6.set_location(np.random.uniform([11, 4.6, 1.5159], [5.5, 10.2, 1.5159]))
    s6.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 6.28319]))


    # Added 8 camera poses via location + euler angles
    
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([25.712, 6.015, 28], [0, 0, 1.5708]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([-17.345, 11.7227, 28], [0, 0, -6.28319]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([-5, 18, 28], [0, 0, 0]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([5.48, 37.4, 28], [0, 0, 1.5708]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([26, 20.4, 28], [0, 0, 0]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([44.194, 13.813, 28], [0, 0, 1.5708]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([6.4, -16, 28], [0, 0, 0]))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([9.7, 5.83, 28], [0, 0, 1.5708]))
    # Render the whole pipeline
    data = bproc.renderer.render()

    # Write data to coco file in a single  output directory Specified in line 9

    bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG",
                                    append_to_existing_output=True)

    bproc.writer.write_hdf5(os.path.join(args.output_dir), data, append_to_existing_output=True)


