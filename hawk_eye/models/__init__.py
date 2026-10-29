def get_model(args, device):
    if args.task == 'ball':
        from hawk_eye.models.ball_detector import get_ball_model
        return get_ball_model(args, device)
    elif args.task == 'court':
        from hawk_eye.models.court_detector import get_court_model
        return get_court_model(args, device)
    else:
        from hawk_eye.models.bounce_detector import get_bounce_model
        return get_bounce_model(args)
