"""Validation Code."""

import torch
from tqdm import tqdm
from collections import defaultdict
from utils.misc import load_pickle, AverageMeter
from models import get_model
from datasets import get_dataloader


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

if __name__ == '__main__':
    args = load_pickle('./model-store/court/ex01/arguments.pkl')
    device = torch.device('cuda:3')

    model, criterion, postprocessor = get_model(args, device)
    model.to(device)
    ckpt = torch.load('./model-store/court/ex01/weights/45_ckpt.pth',
                      weights_only=True,
                      map_location=device)
    print("Load a trained model:", model.load_state_dict(ckpt))
    model.eval()

    args.batch_size = 16
    args.aug_policy = 0
    args.image_set = 'val'
    args.mode = 'val'
    dataloader = get_dataloader(args)
    print(f"Number of images: {len(dataloader.dataset)}")

    error_meter = AverageMeter()
    with torch.inference_mode():
        for inputs, targets in tqdm(dataloader):
            outputs = model(inputs['image'].to(device))

            keypoints, _ = postprocessor(outputs, inputs, refine_flag=0)

            error = criterion.calc_pixel_dist(keypoints, targets)
            error_meter.update(error, n=len(keypoints))
    
    print(f"Error: {error_meter.avg:1.4f}")