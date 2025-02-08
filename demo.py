"""Demo with a image (or a video.)"""

import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from models import get_model
from utils.misc import get_single_image, load_pickle, to_inputs
from utils.court.misc import get_court_img

def demo_model(args, logger):
    device = torch.device(args.device)
    model, _, postprocessor = get_model(args, device)
    ckpt = torch.load(args.model_path,
                      map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt)

    inputs = get_single_image(args)
    with torch.inference_mode():
        outputs = model(inputs['image'].to(device))

        keypoints, _ = postprocessor(outputs, inputs)

    img = inputs['raw_img'][0]
    for point in keypoints[0]:
        cv2.circle(img, list(map(round, point)), 2, (255, 0, 0))

    cv2.imwrite(args.save_root / 'demo.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info("Result save!")

    return None

def video_demo():
    parser = argparse.ArgumentParser()

    parser.add_argument('--court_model_path',
                        type=Path,
                        required=True)
    parser.add_argument('--ball_model_path',
                        type=Path,
                        required=True)
    parser.add_argument('--bounce_model_path',
                        type=Path,
                        required=True)    
    parser.add_argument('--video_path',
                        type=Path,
                        required=True)
    
    args = parser.parse_args()

    device = torch.device('cuda')
    width_minimap = 166
    height_minimap = 350

    # court detection model
    court_args = load_pickle(args.court_model_path.parent.parent / 'arguments.pkl')
    court_model, _, court_postprocessor = get_model(court_args, device)
    court_postprocessor.refine_flag = 3
    imsize= court_args.imsize
    ckpt = torch.load(args.court_model_path,
                      weights_only=True,
                      map_location=device)
    court_model.load_state_dict(ckpt)
    court_model.eval()

    ball_args = load_pickle(args.ball_model_path.parent.parent / 'arguments.pkl')
    ball_model, _, ball_postprocessor = get_model(ball_args, device)
    ckpt = torch.load(args.ball_model_path,
                      weights_only=True,
                      map_location=device)
    ball_model.load_state_dict(ckpt)
    ball_model.eval()

    bounce_args = load_pickle(args.bounce_model_path.parent.parent / 'arguments.pkl')
    _, _, bounce_postprocessor = get_model(bounce_args, device)
    bounce_postprocessor.load_model(args.bounce_model_path)

    # load video frame
    cap = cv2.VideoCapture(args.video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 프레임 속도
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 가로 크기
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 세로 크기

    result = defaultdict(list)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        inputs = to_inputs(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), imsize)
        with torch.inference_mode():
            court_outputs = court_model(inputs['image'].to(device))
            ball_outputs = ball_model(inputs['image'].to(device))

            court_keypoints, court_scores = court_postprocessor(court_outputs, inputs)
            H_inv = cv2.invert(court_postprocessor.matrix)[1]

            ball_keypoints, ball_scores = ball_postprocessor(ball_outputs, inputs)

        ball_keypoints_on_minimap = cv2.perspectiveTransform(np.array(ball_keypoints),
                                                             H_inv)
        result['court'].append((court_keypoints[0], court_scores[0]))
        result['ball'].append((ball_keypoints[0], ball_scores[0]))
        result['ball_minimap'].append(ball_keypoints_on_minimap[0].tolist())
    cap.release()

    x_ball = [item[0][0] for item, _ in result['ball']]
    y_ball = [item[0][1] for item, _ in result['ball']]
    bounce_indices = bounce_postprocessor.predict(x_ball, y_ball)
    
    court_img = get_court_img()
    prev_ball = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('demo_result.mp4', fourcc, fps, (width, height))
    cap = cv2.VideoCapture(args.video_path)
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        court_points, court_scores = result['court'][frame_index]
        ball_points, ball_scores = result['ball'][frame_index]

        for (x, y), score in zip(court_points, court_scores):
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), radius=8, color=(0, 255, 0), thickness=4)
        for (x, y), score in zip(ball_points, ball_scores):
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), radius=8, color=(255, 0, 0), thickness=4)

                if frame_index in bounce_indices:
                    x, y = result['ball_minimap'][frame_index][0]
                    prev_ball.append((int(x), int(y)))
                    color = (255, 0, 0)
                    for i, (x, y) in enumerate(prev_ball):
                        if i == len(prev_ball) - 1:
                            color = (0, 0, 255)
                        cv2.circle(court_img, (x, y), radius=0, color=color, thickness=80)

        court_minimap = cv2.resize(court_img, (width_minimap, height_minimap))
        temp = frame[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :]
        frame[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = temp * 0.2 + court_minimap * 0.8

        video_writer.write(frame)
    cap.release()
    video_writer.release()
    return None

if __name__ == '__main__':
    video_demo()