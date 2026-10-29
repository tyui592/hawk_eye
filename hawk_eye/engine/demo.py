"""Demo with a image (or a video.)"""

import cv2
import torch
import numpy as np
from hawk_eye.models import get_model
from hawk_eye.utils.misc import get_single_image

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

    cv2.imwrite(str(args.save_root / 'demo.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    logger.info("Result save!")

    return None
