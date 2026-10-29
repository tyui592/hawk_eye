"""Court Detector Code."""

from hawk_eye.models.hourglass import get_stacked_hourglass


def get_court_detector(args):
    """Get a model."""
    if args.model == 'hourglass':
        return get_stacked_hourglass(args)
