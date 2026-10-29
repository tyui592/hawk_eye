"""Validation Code."""

import torch
from tqdm import tqdm
from collections import defaultdict
from hawk_eye.utils.misc import AverageMeter
from hawk_eye.models import get_model
from hawk_eye.datasets import get_dataloader


def validate_model(args, logger):
    device = torch.device(args.device)
    model, criterion, postprocessor = get_model(args, device)
    
    ckpt = torch.load(args.model_path, weights_only=True, map_location=device)
    model.load_state_dict(ckpt)
    
    dataloader = get_dataloader(args)
    error, confusion = validate_step(model, dataloader, criterion, postprocessor, device, args)

    logger.info(f"Error: {error:1.4f}")
    recall = confusion['tp'] / (confusion['tp'] + confusion['fn'])
    precision = confusion['tp'] / (confusion['tp'] + confusion['fp'])
    logger.info(f"TP: {confusion['tp']}, FP: {confusion['fp']}, TN: {confusion['tn']}, FN: {confusion['fn']}")
    logger.info(f"Recall: {recall:1.4f}, Precision: {precision:1.4f}")
    return None


def validate_step(model, dataloader, criterion, postprocessor, device, args):
    model.eval()
    
    error_meter = AverageMeter()
    confusion = defaultdict(int)
    with torch.inference_mode():
        for inputs, targets in tqdm(dataloader):
            outputs = model(inputs['image'].to(device))

            keypoints, scores = postprocessor(outputs, inputs)

            error = criterion.calc_pixel_dist(keypoints, scores, targets)
            error_meter.update(error, n=len(keypoints))

            criterion.calc_confusion(confusion, keypoints, scores, targets)

    return error_meter.avg, confusion
