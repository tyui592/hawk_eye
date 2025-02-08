from .detector import get_ball_detector
from .loss import get_criterion
from .postprocessor import get_ball_postprocessor

def get_ball_model(args, device):
    detector = get_ball_detector(args).to(device)

    criterion = get_criterion(args)

    postprocessor = get_ball_postprocessor(args)

    return detector, criterion, postprocessor