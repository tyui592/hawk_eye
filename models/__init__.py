from .court_detector import get_court_model
from .ball_detector import get_ball_model
from .bounce_detector import get_bounce_model

def get_model(args, device):
    if args.task == 'ball':
        return get_ball_model(args, device)
    elif args.task == 'court':
        return get_court_model(args, device)
    else:
        return get_bounce_model(args)