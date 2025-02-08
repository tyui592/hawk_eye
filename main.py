"""Main Code."""
from config import get_arguments
from train import train_model, train_bounce_model
from val import validate_model
from demo import demo_model, video_demo
from utils.misc import get_logger

if __name__ == '__main__':
    args = get_arguments()
    logger = get_logger(args.save_root, args.save_root.stem)
    logger.debug("Arguments")
    for k, v in vars(args).items():
        logger.debug(f"{k}: {v}")

    if args.task in ['ball', 'court']:
        if args.mode == 'train':
            train_model(args, logger)

        elif args.mode == 'val':
            validate_model(args, logger)

    elif args.task == 'bounce' and args.mode == 'train':
        train_bounce_model(args, logger)

    elif args.task == 'demo':
        demo_model(args, logger)
    
    elif args.task == 'video_demo':
        video_demo(args, logger)
