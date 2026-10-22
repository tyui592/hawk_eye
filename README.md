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
- **Classical court detector** (`models/court_detector/classical.py`, no learning).
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

- Scripts
```bash
COURT_DATA="./data/TennisCourtDetector/"
BALL_DATA="./data/TrackNet/"

# Train court detection model
python main.py --mode 'train' --device 'cuda:0' --task 'court' --save_root './model-store/court/' --data_path ${COURT_DATA} --num_class 14 --epoch 100 --lr_milestone 90 --imsize 576 1024 --aug_policy 1

# Validate the classical (non-learning) court detector
python val_classical.py --data_path ${COURT_DATA} --num_workers 8

# Train ball detection model
python main.py --mode 'train' --device 'cuda:1' --task 'ball' --num_class 1 --num_seq 1 --save_root './model-store/ball/' --data_path ${BALL_DATA}  --epoch 100 --lr_milestone 90 --imsize 576 1024 --aug_policy 1

# Train bounce detection model
python main.py --mode 'train' --device 'cpu' --task 'bounce' --num_seq 3 --data_path ${BALL_DATA} --save_root './model-store/bounce/'

# Image demo (court or ball)
python main.py --mode 'demo' --device 'cuda:0' --model_path './model-store/court/weights/ckpt.pth' --image_path './sample.png'

# Video demo
python demo.py --court_model_path './model-store/court/weights/ckpt.pth' --ball_model_path './model-store/ball//weights/ckpt.pth' --bounce_model_path './model-store/bounce/weights/ckpt.cbm' --video_path './sample.mp4'
```


# Result
Recall and Precision (w/ score threshold: 0.5, distance threshold 7 pixels)

| Model | Recall | Precision | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Court | 0.9620 | 0.9997 | 29662 | 8 | 111 | 1173 |
| Court (classical, no learning) | 0.9893 | 0.9997 | 30505 | 9 | 110 | 330 |
| Ball | 0.9216 | 0.9984 | 4372 | 7 | 208 | 372 |
| Bounce | 0.6794 | 0.7607 | 89 | 28 | 4484 | 42 |

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