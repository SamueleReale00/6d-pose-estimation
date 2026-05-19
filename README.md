# 6D Object Pose Estimation — RGB-D Global Fusion Pipeline

**Politecnico di Torino — Advanced Machine Learning**

**Authors:** Francesco Palmisani, Giosuè Pinto, Riccardo Marconi, Samuele Reale

---

## Abstract

6D object pose estimation — recovering a rigid body's full rotation and translation from camera images — is a core enabler for robotic manipulation and AR, yet it remains challenging under texture-less surfaces and occlusion. This project first replicates a strong RGB-only baseline (YOLOv11 + ResNet50 + pinhole translation), then extends it with a global RGB-D fusion architecture that jointly processes color crops and back-projected point clouds, followed by an iterative pose refinement stage. On the LineMOD benchmark the extension raises ADD-0.1d accuracy from 20.0% to 91.6%, approaching the DenseFusion reference at 94.0%.

---

## Pipeline Overview

| Component | Baseline | Extension (Global Fusion) |
|---|---|---|
| Detection | YOLOv11 (bounding box) | YOLOv11-seg (instance masks) |
| RGB branch | ResNet50 | ResNet-18 |
| Depth branch | None | PointNet MLP |
| Translation | Analytical pinhole model | Centroid + learned residual |
| Refinement | None | Iterative PoseRefineNet |
| Loss (rotation) | MSE on quaternion | Point-cloud matching (L1) |
| Loss (translation) | — | L1 on 3D translation |

---

## Architecture Details

### Baseline

**`RotationResNet`** (`models/models_baseline.py`)

ResNet50 with ImageNet pre-training; the final classification head is replaced by a two-layer MLP `Linear(2048→512, ReLU, Dropout 0.1) → Linear(512→4)`. The 4-dimensional output is L2-normalised at forward time to produce a unit quaternion.

**`PinholeTranslationLayer`** (`models/models_baseline.py`)

A parameter-free analytical module. Given the detected bounding box `[x, y, w, h]`, camera intrinsics `K = [fx, fy, cx, cy]`, and the known real-world height of the object model, it estimates depth as:

```
tz = fx * (real_height / w)   if w > h
tz = fy * (real_height / h)   otherwise
```

then back-projects the bounding-box centroid to 3D:

```
tx = (cx_bbox - cx) * tz / fx
ty = (cy_bbox - cy) * tz / fy
```

No learned weights; the entire translation estimate is derived from geometry.

**`BaselinePoseSystem`** (`models/models_baseline.py`)

Inference pipeline: YOLOv11 detection → crop & resize to 224×224 → `RotationResNet` → `PinholeTranslationLayer`. Highest-confidence detection is used; falls back to `(None, None)` on detection failure.

---

### Extension — Global Fusion

**`RGBD_Fusion_Net`** (`models/models_extension.py`)

Two-branch global fusion network:

- **RGB branch:** ResNet-18 backbone (pre-trained), global average pooling → 512-dim feature vector.
- **Depth branch (PointNet MLP):** raw 3D point cloud (N×3, centroid-subtracted) → `Linear(3→64→128)` → per-point max-pooling → `Linear(128→512)` → 512-dim global descriptor.
- **Fusion:** concatenation (1024-dim) → MLP `Linear(1024→512, ReLU) → Linear(512→128, ReLU)`.
- **Heads:** `rot_head: Linear(128→4)` (quaternion) + `trans_head: Linear(128→3)` (translation residual added to centroid).

**`PoseRefineNet`** (`models/models_extension.py`)

1D-convolutional PointNet-style refinement network. Input: point cloud in object-local coordinates (3×N). Architecture: `Conv1d(3→64→128→512→1024)` with ReLU activations → global max-pooling → two independent MLP heads (`Linear(1024→512→128→4)` for Δrotation and `Linear(1024→512→128→3)` for Δtranslation). Applied for `refine_iters` passes (default: 2); each pass transforms the point cloud into the current pose estimate's local frame before predicting the delta.

**`ExtensionPoseSystem`** (`models/models_extension.py`)

Full inference pipeline:
1. YOLOv11-seg produces instance mask + bounding box for the target object.
2. Depth image is masked and back-projected to a 500-point cloud; centroid is subtracted.
3. `RGBD_Fusion_Net` produces coarse quaternion + translation residual; absolute translation = centroid + residual.
4. `PoseRefineNet` iteratively refines the pose: points are transformed into the current local frame, deltas are predicted and applied.

---

## Results

### Main Results (LineMOD validation set, 3,129 samples)

ADD-0.1d metric: a prediction is correct if the mean per-point 3D distance between predicted and ground-truth transformed model points is below 10% of the object diameter (ADD-S used for symmetric objects: eggbox, glue).

| Method | ADD-0.1d |
|---|---|
| **Baseline** (YOLOv11 + ResNet50 + Pinhole) | 20.0% |
| **Extension** (YOLOv11-seg + RGBD Fusion + PoseRefineNet) | 91.6% |
| DenseFusion (reference) | 94.0% |

| Object    | Acc (0.1d) | Acc (2cm) | Err (mm) |
|-----------|------------|-----------|----------|
| Ape       | 91.5%      | 97.3%     | 8.4      |
| Bench     | 97.1%      | 82.4%     | 14.2     |
| Cam       | 67.7%      | 62.1%     | 19.8     |
| Can       | 86.8%      | 70.9%     | 17.4     |
| Cat       | 97.5%      | 98.8%     | 8.0      |
| Drill     | 94.6%      | 78.0%     | 16.0     |
| Duck      | 94.4%      | 98.0%     | 8.6      |
| Egg*      | 100.0%     | 100.0%    | 4.0      |
| Glue*     | 99.6%      | 99.6%     | 4.5      |
| Hole      | 85.0%      | 91.9%     | 11.1     |
| Iron      | 86.1%      | 63.0%     | 19.4     |
| Lamp      | 100.0%     | 91.7%     | 12.0     |
| Phone     | 89.2%      | 75.3%     | 15.3     |
| **AVG**   | **91.6%**  | **85.6%** | **12.1** |
*Symmetric objects (ADD-S metric)

---

### Ablation Study (Baseline pipeline, LineMOD val)

The baseline evaluation script isolates rotation and translation modules independently by providing ground-truth for the complementary component ("oracle" ablation).

| Configuration | ADD-0.1d | Mean ADD | Mean Rot. Err |
|---|---|---|---|
| Rotation only — ResNet50 (GT translation) | 99.0% | 5.7 mm | 7.3° |
| Translation only — Pinhole (GT rotation) | 20.6% | — | — |
| Full baseline (ResNet50 + Pinhole) | 20.0% | — | — |

**Key insight:** The ResNet50 rotation head is highly effective in isolation (99.0% ADD-0.1d with perfect translation). The `PinholeTranslationLayer` is the critical bottleneck: it degrades full-system accuracy from 99% to 20%, because the geometric depth estimate from 2D bounding boxes is insufficiently accurate for tight 10%-diameter thresholds. This diagnosis directly motivates the extension's shift to depth-sensor back-projection for translation.

---

## Dataset

- **LineMOD (Linemod_preprocessed):** 13 texture-less objects in cluttered tabletop scenes.
- **Object IDs covered:** 01–15 (ape, benchvise, bowl, camera, can, cat, cup, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone) — subset availability depends on the preprocessed split used.
- **Split:** deterministic hash-based 80/20 train/val split applied per `(obj_id, frame_idx)` pair using MD5; identical across all dataloaders, ensuring reproducible evaluation.
- **Training samples:** ~12,655 | **Validation samples:** ~3,129
- **Annotations per frame:** 6D pose `(R, t)`, 2D bounding box, binary instance mask (extension only), depth image, camera intrinsics `K`.
- **3D models:** `.ply` mesh files used to compute ADD/ADD-S metrics and for the pinhole height prior.

---

## Installation

```bash
git clone https://github.com/SamueleReale00/6d-pose-estimation
cd 6d-pose-estimation
pip install -r requirements.txt
```

**Requirements summary** (see [requirements.txt](requirements.txt)):
`torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`, `matplotlib`, `pillow`, `plotly`, `ultralytics`, `wandb`, `pyyaml`, `tqdm`, `scikit-learn`, `gdown`, `trimesh`, `scipy`

> All scripts must be run from the **repository root** `6d-pose-estimation/`.

---

## How to Run

### Baseline

```bash
# 1. Download dataset (saves to data/linemod/Linemod_preprocessed/)
python utils/download_dataset.py

# 2. Train YOLO (bounding-box detection)
python scripts/baseline/yolo/yolo_train.py --epochs 100 --batch_size 64

# 3. Train ResNet50 rotation head
python scripts/baseline/resnet/resnet_train.py --epochs 50 --batch_size 64 --lr 0.0001

# 4. Evaluate full baseline pipeline
python scripts/baseline/pipeline_eval.py \
    --yolo_path checkpoints/yolo/best.pt \
    --resnet_path checkpoints/resnet/best/best_*.pth \
    --yolo_conf 0.25

# 5. (Optional) Run single-image inference
python scripts/baseline/pipeline_inference.py \
    --yolo_path checkpoints/yolo/best.pt \
    --resnet_path checkpoints/resnet/best/best_*.pth
```

Use `--help` with any script to see all available arguments.

---

### Extension

```bash
# 1. Download dataset
python utils/download_dataset.py

# 2. Train YOLOv11 segmentation model
python scripts/extension/yolo/yolo_train_seg.py --epochs 100 --batch_size 64

# 3. Train RGB-D fusion network (coarse pose)
python scripts/extension/rgbd_fusion_net/rgbd_fusion_train.py \
    --epochs 50 --batch_size 32 --lr 0.0001

# 4. Train pose refinement network (requires trained fusion net)
python scripts/extension/refine_net/refine_net_train.py \
    --coarse_model_path checkpoints/rgbd_fusion_net/best/best_rgbd_fusion_model.pth \
    --epochs 10 --batch_size 32

# 5. Evaluate full extension pipeline
python scripts/extension/pipeline_eval.py \
    --yolo_path checkpoints/yolo_seg/best.pt \
    --pose_path checkpoints/rgbd_fusion_net/best/best_rgbd_fusion_model.pth \
    --refine_path checkpoints/refine_net/best/best_refine_model.pth

# 6. (Optional) Run single-image inference
python scripts/extension/pipeline_inference.py \
    --yolo_path checkpoints/yolo_seg/best.pt \
    --pose_path checkpoints/rgbd_fusion_net/best/best_rgbd_fusion_model.pth \
    --refine_path checkpoints/refine_net/best/best_refine_model.pth
```

---

## Repository Structure

```
6d-pose-estimation/
├── README.md
├── requirements.txt
├── paper/
│   └── AML_Final_Paper.pdf        # Project research paper
├── notebooks/
│   └── final_pipeline.ipynb       # End-to-end demo notebook (YOLOv11 pipeline)
├── dataset/
│   ├── dataset_baseline.py        # YoloDataset, RotationResNetDataset, BaselineDataset
│   └── dataset_extension.py       # YoloSegDataset, RgbdFusionNetDataset, ExtensionPipelineDataset
├── models/
│   ├── models_baseline.py         # RotationResNet, PinholeTranslationLayer, BaselinePoseSystem
│   ├── models_extension.py        # RGBD_Fusion_Net, PoseRefineNet, ExtensionPoseSystem
│   ├── losses_baseline.py         # QuaternionLoss
│   └── losses_extension.py        # PoseLoss, calc_add_distance
├── scripts/
│   ├── baseline/
│   │   ├── pipeline_eval.py       # Full baseline evaluation (ADD metric, per-object report)
│   │   ├── pipeline_inference.py  # Single-image inference
│   │   ├── resnet/
│   │   │   ├── resnet_train.py    # Train rotation head
│   │   │   ├── resnet_eval.py     # Evaluate rotation head in isolation
│   │   │   └── resnet_inference.py
│   │   └── yolo/
│   │       ├── yolo_train.py      # Fine-tune YOLOv11 for LineMOD detection
│   │       ├── yolo_eval.py
│   │       └── yolo_inference.py
│   └── extension/
│       ├── pipeline_eval.py       # Full extension evaluation
│       ├── pipeline_inference.py  # Single-image inference
│       ├── refine_net/
│       │   ├── refine_net_train.py    # Train PoseRefineNet (requires coarse model)
│       │   ├── refine_net_eval.py
│       │   └── refine_net_inference.py
│       ├── rgbd_fusion_net/
│       │   ├── rgbd_fusion_train.py   # Train coarse RGBD_Fusion_Net
│       │   ├── rgbd_fusion_eval.py
│       │   └── rgbd_fusion_inference.py
│       └── yolo/
│           ├── yolo_train_seg.py      # Fine-tune YOLOv11-seg for LineMOD
│           ├── yolo_eval_seg.py
│           └── yolo_inference_seg.py
└── utils/
    ├── download_dataset.py            # Download LineMOD via gdown from Google Drive
    ├── download_dataset_from_drive.py # Alternative Drive download helper
    ├── evaluation_metrics.py          # ADD, ADD-S, angular error, stats
    ├── process_dataset.py             # Mesh loading, preprocessing helpers
    └── visualization.py               # Pose overlay, point cloud plots
```

---

## Citation

```bibtex
@techreport{palmisani2025pose,
  title     = {6D Object Pose Estimation},
  author    = {Palmisani, Francesco and Pinto, Giosu{\`e} 
               and Marconi, Riccardo and Reale, Samuele},
  institution = {Politecnico di Torino},
  year      = {2025}
}
```
