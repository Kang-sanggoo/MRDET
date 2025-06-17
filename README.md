# Mixture Query Enhanced Transformer for V2X 3D Object Detection
<div align="center">
<a href="https://tum-traffic-dataset.github.io/tumtraf-v2x"><img src="https://img.shields.io/badge/Website-CoopDet3D-0065BD.svg" alt="Website Badge"/></a>
<a href="https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_4"><img src="https://img.shields.io/badge/Dataset-TUMTraf_V2X-0065BD.svg?style=flat&logo=github&logoColor=white" alt="Github Badge"/></a>


## Abstract

Existing V2X-based 3D object detection models primarily employ complex convolutional or recurrent neural network architectures, integrating sensor data from vehicles and infrastructure to perform effective object detection. In this study, we propose a simplified Transformer-based model by adapting the mixture query method introduced in Uni3DETR to the recently proposed CoopDet3D framework. Our model leverages a combination of learnable, non-learnable, and random queries for object detection, utilizing LiDAR and camera data collected from both vehicle and infrastructure perspectives. Experiments conducted on the TUMTraf V2X Cooperative Perception Dataset demonstrated that our proposed model achieves a 3D mAP of 33.04%, precision of 33.38%, and recall of 41.89% on the test set, with a positional RMSE of 0.5208 and rotational RMSE of 0.0478. The results validate the potential of Transformer architectures with mixture query methods in cooperative object detection, contributing foundational insights for the development of perception technologies in autonomous vehicles and intelligent transportation systems (ITS).


## News 📢
- 2025/6: Ranked 4th on the leaderboard (https://eval.ai/web/challenges/challenge-page/2476/leaderboard/6145)

## Dataset Download 📂

1. There are two versions of the [TUMTraf V2X Cooperative Perception Dataset](https://arxiv.org/pdf/2403.01316.pdf) (Release R4) provided:

    1.1. [TUMTraf-V2X](https://innovation-mobility.com/tumtraf-dataset)

    1.2. [TUMTraf-V2X-mini](https://innovation-mobility.com/tumtraf-dataset) (half of the full dataset)

We train CoopDet3D on TUMTraf-V2X-mini and provide the results below.

Simply place the splits in a directory named `tumtraf_v2x_cooperative_perception_dataset` in the `data` directory and you should have a structure similar to this:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_v2x_cooperative_perception_dataset
|   |   ├── train
|   |   ├── val
```

2. The [TUMTraf Intersection Dataset](https://ieeexplore.ieee.org/document/10422289) (Release R2) can be downloaded below:

    2.1 [TUMTraf-I](https://innovation-mobility.com/en/project-providentia/a9-dataset/#anchor_release_2).

Then, download the [TUMTraf Dataset Development Kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit) and follow the steps provided there to split the data into train and val sets.

Finally, place the train and val sets in a directory named `tumtraf_i` in the `data` directory. You should then have a structure similar to this:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_i
|   |   ├── train
|   |   ├── val
```

### Working with python

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
- Latest versions of numba, [torchsparse](https://github.com/mit-han-lab/torchsparse), pypcd, and Open3D

After installing these dependencies, run this command to install the codebase:

```bash
python setup.py develop
```

Finally, you can create a symbolic link `/home/coopdet3d/data/tumtraf_i` to `/home/data/tumtraf_i` and `/home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset` to `/home/data/tumtraf_v2x_cooperative_perception_dataset` in the docker.

### Data Preparation

#### TUMTraf Intersection Dataset

Run this script for data preparation:

```bash
python ./tools/create_tumtraf_data.py --root-path /home/coopdet3d/data/tumtraf_i --out-dir /home/coopdet3d/data/tumtraf_i_processed --splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_i
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_i_processed
│   │   ├── tumtraf_nusc_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── tumtraf_nusc_infos_train.pkl
│   │   ├── tumtraf_nusc_infos_val.pkl
│   │   ├── tumtraf_nusc_dbinfos_train.pkl

```

#### TUMTraf V2X Cooperative Perception Dataset

Run this script for data preparation:

```bash
python ./tools/create_tumtraf_v2x_data.py --root-path /home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset --out-dir /home/coopdet3d/data/tumtraf_v2x_cooperative_perception_dataset_processed --splits training,validation
```

After data preparation, you will be able to see the following directory structure:

```
coopdet3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_v2x_cooperative_perception_dataset
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_v2x_cooperative_perception_dataset_processed
│   │   ├── tumtraf_v2x_nusc_gt_database
|   |   ├── train
|   |   ├── val
│   │   ├── tumtraf_v2x_nusc_infos_train.pkl
│   │   ├── tumtraf_v2x_nusc_infos_val.pkl
│   │   ├── tumtraf_v2x_nusc_dbinfos_train.pkl

```

### Training

**NOTE 1:** If you want to use a YOLOv8 `.pth` file from MMYOLO, please make sure the keys inside fit with this model. Convert that `.pth` checkpoint using this converter: `./tools/convert_yolo_checkpoint.py`. 

**Note 2:** The paths to the pre-trained weights for YOLOv8 models are hardcoded in the config file, so change it there accordingly. This also means that when training models that use YOLOv8, the parameters `--model.encoders.camera.backbone.init_cfg.checkpoint`, `--model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint`, and `--model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint` are optional.

**Note 3:** We trained our model on 3 GPUs (3 x RTX 3090) and used the following prefix for that: `torchpack dist-run -np 3` 

For training a camera-only model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/centerhead/lssfpn/camera/256x704/yolov8/default.yaml
```

For training LiDAR-only model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE>
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/transfusion/secfpn/lidar/pointpillars.yaml
```

For training a fusion model on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```

Example:

```
torchpack dist-run -np 3 python tools/train.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --load_from weights/coopdet3d_tumtraf_i_l_pointpillars512_2x.pth
```

For training camera-only model on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> --model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both).


Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/centerhead/lssfpn/cooperative/camera/256x704/yolov8/default.yaml
```

For training LiDAR-only model on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE>
```

Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/lidar/pointpillars.yaml
```

For training fusion model on the TUMTraf V2X Cooperative Perception Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> ---model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```
Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both).

Example:

```
torchpack dist-run -np 3 python tools/train_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --load_from weights/coopdet3d_vi_l_pointpillars512_2x.pth
```

Note: please run `tools/test.py` or `tools/test_coop.py` separately after training to get the final evaluation metrics.

### BEV mAP Evaluation (Customized nuScenes Protocol)

**NOTE: This section will not work without the test set ground truth, which is not made public. To evaluate your model's mAP<sub>BEV</sub>, please send your config files and weights to the authors for evaluation!**

For evaluation on the TUMTraf Intersection Dataset, run:

```
torchpack dist-run -np 1 python tools/test.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

Example:

```
torchpack dist-run -np 1 python tools/test.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --eval bbox
```

For evaluation on the TUMTraf V2X Cooperative Perception Dataset, run:

```
torchpack dist-run -np 1 python tools/test_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

Example:

```
torchpack dist-run -np 1 python tools/test_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --eval bbox
```

### Running CoopDet3D Inference and Save Detections in OpenLABEL Format

Exporting to OpenLABEL format is needed to perform mAP<sub>3D</sub> evaluation or detection visualization using the scripts in the [TUM Traffic dev-kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit).

**NOTE: You will not be evaluate your inference results using the dev-kit without the test set ground truth, which is not made public. To evaluate your model's mAP<sub>3D</sub>, please send your detection results to the authors for evaluation!**

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/inference_to_openlabel.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --out-dir <PATH_TO_OPENLABEL_OUTPUT_FOLDER>
```

Example:

```
torchpack dist-run -np 1 python tools/inference_to_openlabel.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --out-dir inference
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_CHECKPOINT_PTH> --split [train, val, test] --input_type hard_drive --save_detections_openlabel --output_folder_path_detections <PATH_TO_OPENLABEL_OUTPUT_FOLDER>
```

Example:
```
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/bevfusion_coop_vi_cl_pointpillars512_2x_yolos.pth --split test --input_type hard_drive --save_detections_openlabel --output_folder_path_detections inference
```

### Runtime Evaluation:

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/benchmark.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 50
```

Example:

```
torchpack dist-run -np 1 python tools/benchmark.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --log-interval 50
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python tools/benchmark_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 10
```

Example:

```
torchpack dist-run -np 1 python tools/benchmark_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --log-interval 10
```

### Built in visualization:

For TUMTraf Intersection Dataset:

```
torchpack dist-run -np 1 python tools/visualize.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_tumtraf 
```

Example:

```
torchpack dist-run -np 1 python tools/visualize.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_tumtraf_i_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --mode pred --out-dir viz_tumtraf 
```

For TUMTraf V2X Cooperative Perception Dataset:

```
torchpack dist-run -np 1 python tools/visualize_coop.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_tumtraf 
```

Example:

```
torchpack dist-run -np 1 python tools/visualize_coop.py configs/tumtraf_v2x/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/coopdet3d_vi_cl_pointpillars512_2x_yolos_transfer_learning_best.pth --split test --mode pred --out-dir viz_tumtraf 
```

For split, naturally one could also choose "train" or "val". For mode, the other options are "gt" (ground truth) or "combo" (prediction and ground truth).

**NOTE: Ground truth visualization on test set will not work since the test set provided is missing the ground truth.**


## Acknowledgement 🤝 

The codebase is built upon [BEVFusion](https://github.com/mit-han-lab/bevfusion) with vehicle-infrastructure fusion inspired by the method proposed in [PillarGrid](https://arxiv.org/pdf/2203.06319.pdf).


