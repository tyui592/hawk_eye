def get_court_model(args, device):
    from hawk_eye.models.court_detector.detector import get_court_detector
    detector = get_court_detector(args).to(device)

    from hawk_eye.models.court_detector.loss import get_criterion
    criterion = get_criterion(args)

    from hawk_eye.models.court_detector.postprocessor import get_court_postprocessor
    postprocessor = get_court_postprocessor(args)

    return detector, criterion, postprocessor
