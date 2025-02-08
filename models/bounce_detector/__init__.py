from .detector import get_bounce_detector
from .postprocessor import BouncePostprocessor

def get_bounce_model(args):
    detector = get_bounce_detector(args)
    postprocessor = BouncePostprocessor(num_frame=args.num_seq)
    return detector, None, postprocessor