from .detector import get_court_detector
from .loss import get_criterion
from .postprocessor import get_court_postprocessor

def get_court_model(args, device):
    detector = get_court_detector(args).to(device)

    criterion = get_criterion(args)

    postprocessor = get_court_postprocessor(args)

    return detector, criterion, postprocessor