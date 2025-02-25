# blender_proc_experiments

In this repo we use BlenderProc to create synthetic data for object detection.

[coco2yolo.py](coco2yolo.py) converts coco format to YOLO.

[create_synthetic_data.py](./coco2yolo.py) creates the synthetic data by performing renderings in the given scene using specific camera and moving specific objects.

[filter_coco_annotations.py](./create_synthetic_data.py) filters the categories to only the desired ones, since the models contains a big variety of category IDs.

[verify_dataset.py](./verify_dataset.py) verifies the yolo-transformed dataset



## BlenderProc-related commands
```
blenderproc run create_synthetic_data.py
blenderproc vis hdf5 output/0.hdf5
blenderproc vis coco -i 0 -c coco_annotations.json -b output/coco_data
```


##  Required packages
The current repo has been tested in win 11, using BlenderProc 2.8.0


## Author (or who to blame)
Angelos Plastropoulos