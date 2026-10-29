def get_ball_model(args, device):
    from hawk_eye.models.ball_detector.detector import get_ball_detector
    detector = get_ball_detector(args).to(device)

    from hawk_eye.models.ball_detector.loss import get_criterion
    criterion = get_criterion(args)

    from hawk_eye.models.ball_detector.postprocessor import get_ball_postprocessor
    postprocessor = get_ball_postprocessor(args)

    return detector, criterion, postprocessor
