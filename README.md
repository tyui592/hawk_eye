Hawk-Eye
==

# Objectives
1. Detect tennis court edges.
2. Detect tennis ball and bounces.
3. (_Ongoing_) Estimate tennis player poses.

# Datasets
- [yastrebksv's dataset](https://github.com/yastrebksv/TennisCourtDetector) for court edge detection.
- [TrackNet](https://arxiv.org/abs/1907.03698) dataset for ball and bounce detection.

# Methods
- **Stacked Hourglass Network** for court and ball detection.
  - Use **focal loss** for heatmap training. (Reference: _Object as Points_, arXiv 2019.)
  - Perform data augmentation to enhance robustness against varying camera angles.
  ![data_augmentation](./assets/tennis_data_augmentation.png)
  - Refining the edge points of a tennis court using image processing.
  ![line](./assets/line_processing.png)
- **Classical court detector** (`hawk_eye/models/court_detector/classical.py`, no learning).
  - White line pixels by a relative local contrast test (robust to shadows) + structure tensor coherence filter.
  - Hough lines, homography hypotheses from 2 horizontal + 2 vertical line pairs, scored by court model overlap and penalized for lines where the court has none (e.g. net tape).
  - Reject physically implausible homographies by recovering the camera (focal length, height, position).
  - Refine the homography by re-fitting court lines to nearby line pixels.
  - Step-by-step walkthrough with intermediate results: [notebooks/classical_court_detection.ipynb](./notebooks/classical_court_detection.ipynb)
- **Catboost** for ball bounce detection.

# Usage
- Requirements
  - catboost
  - torch
  - torchvision
  - albumentations
  - opencv-python
  - numpy

- Scripts (run from the repository root; no installation step)
```bash
COURT_DATA="./data/TennisCourtDetector/"
BALL_DATA="./data/TrackNet/"

# Train court detection model
python -m hawk_eye.cli.main --mode 'train' --device 'cuda:0' --task 'court' --save_root './model-store/court/' --data_path ${COURT_DATA} --num_class 14 --epoch 100 --lr_milestone 90 --imsize 576 1024 --aug_policy 1

# Validate the classical (non-learning) court detector
python -m hawk_eye.cli.eval_court --data_path ${COURT_DATA} --num_workers 8

# Create the separately reviewed-label dataset, then evaluate train + val
python -m hawk_eye.cli.prepare_court_dataset
python -m hawk_eye.cli.eval_court --data_path ./data/TennisCourtDetector_v3_reviewed \
  --image_set all --num_workers 8 --save_vis fail \
  --save_dir ./data/outputs/court_detection/classical-v3/tenniscourt-all-reviewed-new

# Train ball detection model
python -m hawk_eye.cli.main --mode 'train' --device 'cuda:1' --task 'ball' --num_class 1 --num_seq 1 --save_root './model-store/ball/' --data_path ${BALL_DATA}  --epoch 100 --lr_milestone 90 --imsize 576 1024 --aug_policy 1

# Train bounce detection model
python -m hawk_eye.cli.main --mode 'train' --device 'cpu' --task 'bounce' --num_seq 3 --data_path ${BALL_DATA} --save_root './model-store/bounce/'

# Image demo (court or ball)
python -m hawk_eye.cli.main --mode 'demo' --device 'cuda:0' --data_path "${COURT_DATA}" --model_path './model-store/court/weights/ckpt.pth' --image_path './sample.png'

# Video demo
python -m hawk_eye.cli.demo --court_model_path './model-store/court/weights/ckpt.pth' --ball_model_path './model-store/ball//weights/ckpt.pth' --bounce_model_path './model-store/bounce/weights/ckpt.cbm' --video_path './sample.mp4'

# DL validation (loads model arguments from the checkpoint's experiment)
python -m hawk_eye.cli.main --mode val --task court --data_path "${COURT_DATA}" --model_path './model-store/court/weights/ckpt.pth'

# Classical ball evaluation and court-gated visualization
python -m hawk_eye.cli.eval_ball --data_path "${BALL_DATA}" --clips game10/Clip1,game10/Clip6
python -m hawk_eye.cli.visualize_ball --data_path "${BALL_DATA}" --clip game10/Clip1

# Inspect historical output organization without modifying files
python -m hawk_eye.cli.organize_outputs
```

Code lives in `hawk_eye/`: `cli` handles commands, `engine` runs training and DL
evaluation, `evaluation` handles classical metrics and visualization, and `pipelines`
connects video models. Import classes through `hawk_eye.models.…`; legacy root
modules are removed. Data, weights and notebook outputs remain in their existing
locations. Checkpoint state keys and `arguments.pkl` are unchanged.


# Result

Local experiment records and versioned artifacts: [experiment index](./docs/experiments/README.md)
and [output catalog](./data/outputs/README.md). Results are grouped by `court_detection`,
`ball_tracking`, and `pipeline`, then method/version and run. These docs and artifacts
are local-only under the repository's current Git policy.

Recall and Precision (w/ score threshold: 0.5, distance threshold 7 pixels)

| Model | Evaluation split / labels | Recall | Precision | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Court (DL Hourglass) | val 2,211 / original | 0.9620 | 0.9997 | 29662 | 8 | 111 | 1173 |
| Court (classical v3, no learning) | val 2,211 / original | 0.989298 | 0.999705 | 30505 | 9 | 110 | 330 |
| Court (classical v3, no learning) | train 6,630 / original | 0.988632 | 0.999891 | 91486 | 10 | 272 | 1052 |
| Court (classical v3, no learning) | train + val 8,841 / original | 0.988798 | 0.999844 | 121991 | 19 | 382 | 1382 |
| Court (classical v3, no learning) | val 2,211 / v3-reviewed | 1.000000 | 1.000000 | 30843 | 0 | 111 | 0 |
| Court (classical v3, no learning) | train + val 8,841 / v3-reviewed | 0.991474 | 0.999918 | 122329 | 10 | 383 | 1052 |
| Ball | original validation baseline | 0.9216 | 0.9984 | 4372 | 7 | 208 | 372 |
| Bounce | original validation baseline | 0.6794 | 0.7607 | 89 | 28 | 4484 | 42 |

The owner reviewed all 102 classical-v3 val failures and judged the original GT
incorrect. A separate dataset, `data/TennisCourtDetector_v3_reviewed`, replaces
those images' complete 14-point labels with the approved v3 projections; the
original dataset and all train labels remain unchanged. The reviewed-val result
measures agreement with these v3-derived labels, not independent accuracy.

The original train + val row combines saved train and original-val confusion
counts; both val runs produced identical detections. DL was evaluated only on the
original val labels, so compare it with the classical original-val row. Classical
v3 needs no training, but its parameters were developed using val; the expanded
evaluation is not an external test. The 295 train mismatches await manual review.
See the [court experiment record](./docs/experiments/court_detection.md#ct-004-결과)
and [saved evaluation](./data/outputs/court_detection/classical-v3/tenniscourt-all-v3-reviewed-labels/RUN.md).

Demo video

![demo](./assets/demo.gif)

# Todo
- [x] Court edge detection
- [x] Ball detection and bounce detection
- [ ] Player pose estimation
- [ ] Match summary (e.g., ball speed, ...)
- [ ] Improve the court edge detection algorithm
  - [ ] Utilize more recent keypoint detection models
    - [ ] _Deep High-Resolution Representation Learning for Human Pose Estimation_, 2019 CVPR
    - [ ] _Sapiens: Foundation for Human Vision Models_, 2024 ECCV
  - [ ] Enhance the post-processing algorithm
    - [ ] Improve an image processing-based edge detection algorithm
- [ ] Improve the ball detection algorithm
- [ ] Improve the bounce detection algorithm
- [ ] Develop a unified model for court, ball detection, and human pose estimation

# Reference
- TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications [2019, arXiv](https://arxiv.org/pdf/1907.03698)
- https://github.com/yastrebksv/TennisProject
