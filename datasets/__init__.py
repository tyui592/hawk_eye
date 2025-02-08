from .ball import get_ball_dataloader
from .ball_seq import get_ballseq_dataloader
from .court import get_court_dataloader
from .bounce import get_bounce_datasets

def get_dataloader(args):
    if args.task == 'ball':
        if args.num_seq > 1:
            return get_ballseq_dataloader(args)
        else:
            return get_ball_dataloader(args)

    elif args.task == 'court':
        return get_court_dataloader(args)

    else:
        return get_bounce_datasets(args)