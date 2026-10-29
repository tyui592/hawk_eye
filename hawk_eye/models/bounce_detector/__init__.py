def get_bounce_model(args):
    from hawk_eye.models.bounce_detector.detector import get_bounce_detector
    detector = get_bounce_detector(args)
    from hawk_eye.models.bounce_detector.postprocessor import BouncePostprocessor
    postprocessor = BouncePostprocessor(num_frame=args.num_seq)
    return detector, None, postprocessor
