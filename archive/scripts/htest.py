import blenderproc as bproc
import argparse
import numpy as np
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('scene', nargs='?', default="C:/Users/Bende/Documents/blender_hangar/scenes/a320_in_hangar_3.blend", help="Path to the scene.blend file")
parser.add_argument('output_dir', nargs='?', default="output/", help="Path to where the final files will be saved")
parser.add_argument('--runs', type=int, default=1, help="Number of runs to perform")
args = parser.parse_args()

bproc.init()

# Load the objects into the scene
objs = bproc.loader.load_blend(args.scene)

# Define objects and their respective category IDs
object_categories = {
    1: ["air_conduit", "ceiling", "ceiling.001", "ceiling.002", "ceiling.003",
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
        "windows_offices"],
    2: ["PROP_STILL_LEFT", "PROP_STILL_RIGHT", "REACTOR_BACK_LEFT_1", "REACTOR_BACK_RIGHT_1",
         "REACTOR_LEFT", "REACTOR_RIGHT"],
    3: ["AILERON_LEFT", "AILERON_RIGHT", "FLAPS_01_LEFT", "FLAPS_01_RIGHT", "FLAPS_02_LEFT", 
         "FLAPS_02_RIGHT", "FLAPSKRUEGER_02_LEFT", "FLAPSKRUEGER_02_RIGHT", "FLAPSKRUEGER_LEFT", 
         "FLAPSKRUEGER_RIGHT", "SPOILER_2_1_LEFT", "SPOILER_2_1_RIGHT", "SPOILER_2_3_LEFT", 
         "SPOILER_2_3_RIGHT", "SPOILER_LEFT", "SPOILER_RIGHT", "TAIL_ELEVATOR_LEFT", 
         "TAIL_ELEVATOR_RIGHT", "TAIL_RUDDER"],
    4: ["CH_Moving", "CH_Static", "DECAL_BANDIT", "DOOR", "DOOR_PASSENGER", "DOOR_REAR", "FUSELAGE",
         "FUSELAGE.FINS_EXTERNAL.DETAILS", "FUSELAGE.WHEELBAYS", "HoneywellJetwave", "LIVERY_OFFICIAL_FUSELAGE",
         "LIVERY_OFFICIAL_FUSELAGE.RED DETAILS", "LIVERY_OFFICIAL_TRIML", "LIVERY_OFFICIAL_TRIMR",
         "TAIL_ELEVATOR_TRIM_LEFT", "TAIL_ELEVATOR_TRIM_RIGHT", "WIPER_BASE_L_1", "WIPER_BASE_R_1",
         "WIPER_WIPER_L_1", "WIPER_WIPER_R_1"],
    5: ["C_DOOR_01_HYDROLIC_LEFT", "C_DOOR_01_HYDROLIC_RIGHT", "C_DOOR_01_LEFT", "C_DOOR_01_RIGHT",
         "C_DOOR_02_HYDROLIC_LEFT", "C_DOOR_02_HYDROLIC_RIGHT", "C_DOOR_02_LEFT", "C_DOOR_02_RIGHT", 
         "DOOR01_LEFT", "DOOR01_RIGHT", "DOOR02_LEFT", "DOOR02_RIGHT", "DOOR03_HYDROLIC1_LEFT", 
         "DOOR03_HYDROLIC2_LEFT", "DOOR03_LEFT", "R_DOOR03_HYDROLIC1_RIGHT", "R_DOOR03_HYDROLIC_RIGHT", 
         "R_DOOR03_RIGHT"],
    6: ["FLAPSFAIRING_04_LEFT_1", "FLAPSFAIRING_04_RIGHT_1", "FLAPSFAIRING_05_LEFT_1",
         "FLAPSFAIRING_05_RIGHT_1", "FLAPSFAIRING_06_LEFT_1", "FLAPSFAIRING_06_RIGHT_1",
         "SHARKLET_LIVERY_OFFICIAL_WINGL", "SHARKLET_LIVERY_OFFICIAL_WINGL", "WING_LEFT_1", "WING_RIGHT_1"],
    7: ["Electric Lift Truck.001"],
    8: ["Object_2", "Object_3", "Object_4", "Object_5", "Object_6", "Object_7", "Object_8",
        "Object_9", "Object_10", "Object_11", "Object_12", "Object_13", "Object_14",
        "Object_15", "Object_16", "Object_17", "Object_18", "Object_19", "Object_20",
        "Object_21", "Object_22"],
    9: ["Drone"],
    10: ["Tool cart"],
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
K_matrix = np.array([[2666.6667, 0.0000, 960.0000],
                     [0.0000, 2666.6667, 540.0000],
                     [0.0000, 0.0000, 1.0000]])

bproc.camera.set_intrinsics_from_K_matrix(K=K_matrix, image_width=1920, image_height=1080)

# activate normal rendering
# bproc.renderer.enable_normals_output()
# bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# Perform multiple runs
for r in range(args.runs):
    # Clear all keyframes from the previous run
    bproc.utility.reset_keyframes()

    # Place the objects randomly for each run
    f1 = bproc.filter.one_by_attr(objs, "name", "Drone")

    f1.set_location(np.random.uniform([-38.0, 0.0, 3.0], [-34.0, 8.0, 3.0]))
    f1.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, 1.5708]))

    # Added camera pose via location + euler angles
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([-36.0, 4.0, 18.0], [0, 0, 1.5708]))
    
    # Render the whole pipeline
    data = bproc.renderer.render()

    # Write data to coco file in a single  output directory Specified in line 9
    # bproc.writer.write_coco_annotations(os.path.join(args.output_dir, 'coco_data'),
    #                                 instance_segmaps=data["instance_segmaps"],
    #                                 instance_attribute_maps=data["instance_attribute_maps"],
    #                                 colors=data["colors"],
    #                                 color_file_format="JPEG",
    #                                 append_to_existing_output=True)

    # bproc.writer.write_hdf5(os.path.join(args.output_dir), data, append_to_existing_output=True)
    bproc.writer.write_hdf5(os.path.join(args.output_dir), data)