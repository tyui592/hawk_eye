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

# 사용법

## 1. 실행 환경 준비

모든 명령은 저장소 루트에서 실행한다. 패키지를 별도로 설치하지 않고
`python -m hawk_eye.cli.<명령>` 형식을 사용한다.

- **Mac / CPU**: classical 코트·공 검출, 평가, 영상 시각화, 노트북.
- **CUDA 머신**: Hourglass 학습·평가와 DL 영상 데모.
- **CatBoost 바운스 학습**: 아래 예시는 CPU를 사용한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python scipy tqdm torch torchvision \
  catboost albumentations pandas scikit-learn matplotlib
python -c "import cv2, torch; print('OpenCV:', cv2.__version__, 'PyTorch:', torch.__version__)"
```

CUDA 머신에서는 먼저 해당 머신의 CUDA 환경에 맞는 PyTorch·torchvision을 준비한다.
위 명령은 의존성 설치 예시이며 버전을 고정한 재현 환경 파일은 아니다.
노트북을 사용할 때는 `python -m pip install ipykernel jupyter`도 실행하고
`.venv/bin/python`을 커널로 선택한다.

현재 로컬 체크아웃의 기존 `.venv`는 인터프리터 경로가 깨져 있다.
패키지 재편 검증에는 별도 `scratch/package-venv`를 사용했다.
기존 환경을 복구하기 전 이 체크아웃에서 명령을 확인하려면
`source scratch/package-venv/bin/activate`로 해당 환경을 사용할 수 있다.

## 2. 데이터와 가중치 준비

데이터와 학습 가중치는 별도로 준비한다. Git에는 포함되지 않는다.

```text
data/
├── TennisCourtDetector/
│   ├── data_train.json
│   ├── data_val.json
│   └── images/<image-id>.png
└── TrackNet/
    └── game10/Clip1/
        ├── Label.csv
        ├── 0000.jpg
        └── ...
model-store/
└── court/<실험명>/
    ├── arguments.pkl
    └── weights/ckpt.pth
```

- 코트: [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)의 이미지와 JSON 라벨.
- classical 공 평가·영상 및 바운스 학습: TrackNet의 `game*/Clip*/Label.csv`와 프레임 이미지.
- DL 공 학습·평가: 단일 프레임은 `data_train.pkl` / `data_val.pkl`,
  시퀀스는 `data_train_clip.pkl` / `data_val_clip.pkl`이 추가로 필요하다.
  원본 TrackNet 폴더만으로 이 파일들이 자동 생성되지는 않는다.

```bash
COURT_DATA="./data/TennisCourtDetector"
BALL_DATA="./data/TrackNet"
```

명시한 상대 경로는 현재 작업 디렉터리 기준이다.
`--save_dir`에는 실행마다 새 폴더명을 사용한다.

## 3. 빠른 시작: classical 검출

가중치 없이 실행할 수 있다. 아래 코트 명령은 val 전체를 평가하며,
공 명령은 두 클립만 평가한다.

```bash
# 코트: val 평가 및 실패 이미지만 저장
python -m hawk_eye.cli.eval_court \
  --data_path "${COURT_DATA}" --image_set val --num_workers 8 \
  --save_vis fail --save_dir ./data/outputs/court_detection/classical-v3/my-val-run

# 공: game10 Clip1·Clip6만 평가
python -m hawk_eye.cli.eval_ball \
  --data_path "${BALL_DATA}" --clips game10/Clip1,game10/Clip6 --num_workers 2 \
  --save_dir ./data/outputs/ball_tracking/classical-v1b/my-two-clips-run
```

코트의 `--image_set`은 `train`, `val`, `all` 중 선택한다.
`--save_vis`는 `all`, `fail`, `none`을 지원한다.
공의 `--clips`를 생략하면 game9·game10 전체를 평가한다.
두 평가 모두 `--save_dir`를 생략하면 결과 파일을 저장하지 않고 지표를 출력한다.

## 4. 공 추적 영상 만들기

기본값은 코트가 검출된 장면에서만 공 추적을 허용하는 방식이다.

```bash
# TrackNet 클립
python -m hawk_eye.cli.visualize_ball \
  --data_path "${BALL_DATA}" --clip game10/Clip1 --show_cands

# 일반 영상: 먼저 300프레임으로 확인
python -m hawk_eye.cli.visualize_ball \
  --video_path ./data/DemoVideos/0.mp4 --max_frames 300

# 이동 제한 옵션을 켜고 기본 출력과 좌우 비교
python -m hawk_eye.cli.visualize_ball \
  --video_path ./data/DemoVideos/0.mp4 --max_frames 300 \
  --params '{"motion_prior":true}' --compare_motion
```

`--max_frames`를 생략하면 영상 끝까지 처리한다.
`--clip`과 `--video_path` 중 하나만 지정한다.
`--no_court_gate`는 코트 판정을 끄고,
`--court_interval 1`은 매 프레임 코트를 재검출한다.
기본 재검출 주기는 약 1초이며 장면 전환에서도 다시 확인한다.
`motion_prior`(프레임 간 이동 제한)는 recall 감소가 있어 기본값이 `false`다.
`--compare_motion`은 일반 영상 입력과 `motion_prior=true`를 함께 지정해야 한다.

## 5. DL·바운스 모델 학습

실험마다 새 `--save_root`를 사용한다. 학습 시 `arguments.pkl`,
`weights/`, `log.txt`가 해당 폴더에 저장된다.

```bash
# CUDA: 코트 Hourglass
python -m hawk_eye.cli.main --mode train --task court --device cuda:0 \
  --data_path "${COURT_DATA}" --save_root ./model-store/court/my-run \
  --num_class 14 --epoch 100 --lr_milestone 90 --imsize 576 1024 --aug_policy 1

# CUDA: 공 Hourglass (data_train.pkl / data_val.pkl 필요)
python -m hawk_eye.cli.main --mode train --task ball --device cuda:0 \
  --data_path "${BALL_DATA}" --save_root ./model-store/ball/my-run \
  --num_class 1 --num_seq 1 --epoch 100 --lr_milestone 90 \
  --imsize 576 1024 --aug_policy 1

# CPU: CatBoost 바운스
python -m hawk_eye.cli.main --mode train --task bounce --device cpu \
  --data_path "${BALL_DATA}" --save_root ./model-store/bounce/my-run --num_seq 3
```

`--imsize`의 순서는 높이·너비다. 위 예시의 `576 1024`는
높이 576px, 너비 1024px를 뜻한다.
바운스 학습 결과는 `weights/ckpt.cbm`에 저장된다.

## 6. DL 평가와 이미지 데모

`--model_path`는 가중치 파일을 가리킨다.
모델 구성은 가중치 폴더의 상위 실험 폴더에 있는 `arguments.pkl`에서 복원하므로
가중치 파일과 이 파일을 함께 보관한다.

```bash
# 코트 DL val 평가
python -m hawk_eye.cli.main --mode val --device cuda:0 \
  --data_path "${COURT_DATA}" --model_path ./model-store/court/my-run/weights/ckpt.pth

# 이미지 데모
python -m hawk_eye.cli.main --mode demo --device cuda:0 \
  --data_path "${COURT_DATA}" --model_path ./model-store/court/my-run/weights/ckpt.pth \
  --image_path ./sample.png
```

`--data_path`는 이미지 데모에서도 필수 인자다.
공 모델도 해당 실험의 가중치와 데이터 경로로 바꿔 실행한다.
기본 출력은 실험 폴더의 `val/` 또는 `demo/`이며,
이미지 데모는 `demo.jpg`를 저장한다. `--save_root`로 출력 위치를 지정할 수 있다.

DL 영상 데모의 진입점은 다음과 같다.

```bash
python -m hawk_eye.cli.demo \
  --court_model_path ./model-store/court/my-run/weights/ckpt.pth \
  --ball_model_path ./model-store/ball/my-run/weights/ckpt.pth \
  --bounce_model_path ./model-store/bounce/my-run/weights/ckpt.cbm \
  --video_path ./sample.mp4
```

이 경로는 CUDA를 하드코딩하고 현재 작업 디렉터리의 `demo_result.mp4`로 저장한다.
좌표 스케일과 영상 루프 등 기존 동작의 검증이 남아 있는 실험 경로다.
이번 패키지 재편에서는 CUDA 학습·DL 실추론을 검증하지 않았다.

## 7. 출력과 평가 지표

| 실행 | 저장 항목 |
| --- | --- |
| classical 코트 평가 | `results.json`, `summary.csv`, `fail_triage.csv`, `metrics.txt`, 선택한 `vis/`, `manifest.json` |
| classical 공 평가 | `results.json`, `summary.csv`, `metrics.txt`, `manifest.json` |
| 공 추적 영상 | MP4, `manifest.json`; 일반 영상은 `.summary.json`, 코트 게이트 사용 시 `.court.json`도 저장 |
| DL 학습·평가 | 실험 폴더의 인자·가중치·로그 및 모드별 출력 |

영상의 자동 출력은 `data/outputs/<목표>/<방법-버전>/demo-<시각>/`다.
코트 게이트 사용 시 목표는 `pipeline`, 끄면 `ball_tracking`이다.
classical 실행의 명시 출력 폴더는 이미 존재하면 거부한다.
`manifest.json`은 실행 설정·코드 revision·명령·시각·완료 상태를 기록한다.
이 보호 규칙은 DL의 `--save_root`와 별개다.

GT는 정답 라벨, TP는 정답 검출, FN은 정답 누락, FP는 정답이 없는 곳의 검출을 뜻한다.
코트·공의 TP는 **score > 0.5, GT와 거리 < 7px**를 모두 만족해야 한다.
정확히 0.5 또는 7px인 경우는 TP가 아니다.
가시 GT가 있는 프레임에서 위치가 틀리면 FN으로 집계하며,
GT가 없을 때 검출하면 FP로 집계한다.
따라서 precision만으로 모든 위치 오검출을 설명할 수 없다.

호모그래피는 코트 평면의 좌표 변환이다.
공의 코트 좌표·속도 추정에는 **z=0인 바운스 지점에서만** 적용한다.
공중 공의 이미지 좌표를 그대로 지면 좌표로 변환해 속도를 계산하지 않는다.

## 8. 보조 명령과 확인

```bash
# 검토한 102개 val 라벨을 별도 데이터셋에 반영 (기존 목적지 거부)
python -m hawk_eye.cli.prepare_court_dataset \
  --destination ./data/TennisCourtDetector_v3_reviewed

# 결과 폴더 이관 계획만 확인 (실제 이관은 --apply)
python -m hawk_eye.cli.organize_outputs

# 옵션과 기본값 확인
python -m hawk_eye.cli.eval_court --help
python -m hawk_eye.cli.visualize_ball --help
python -m hawk_eye.cli.main --help

# 회귀 테스트
python -m unittest discover -s tests
```

라벨 보정 도구는 이 프로젝트에서 이미 검토한 v3 결과 파일을 입력으로 사용한다.
임의 데이터셋의 라벨을 자동으로 보정하는 범용 도구가 아니다.
검토 라벨과 비교한 점수는 독립 테스트 정확도와 구분해야 한다.

## 코드 구조

```text
hawk_eye/
├── cli/          # 인자 처리와 명령 실행
├── engine/       # 학습·DL 평가·이미지 데모
├── evaluation/   # classical 평가·판정·집계·영상 시각화
├── pipelines/    # 코트 게이트·구간 추적·DL 영상 데모
├── models/       # classical 검출기·Hourglass·CatBoost
├── datasets/     # 학습 데이터 로더·검토 데이터셋 생성
├── experiments/  # 실행 메타데이터·결과 폴더 관리
├── utils/        # I/O·히트맵·코트 기하
├── config.py
└── paths.py      # 저장소 루트 경로
```

내부 import는 `from hawk_eye.models.court_detector.classical import ClassicalCourtDetector`처럼 쓴다.
이전 루트 `main.py`, `models`, `datasets`, `utils` 경로의 호환 래퍼는 없다.
데이터·가중치·노트북은 기존 위치에 유지한다.


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
