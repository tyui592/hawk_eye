"""Main Code."""
from hawk_eye.config import get_arguments
from hawk_eye.utils.misc import get_logger

def main():
    args = get_arguments()
    from hawk_eye.engine.train import train_model, train_bounce_model
    from hawk_eye.engine.validate import validate_model
    from hawk_eye.engine.demo import demo_model

    logger = get_logger(args.save_root, args.save_root.stem)
    logger.debug("Arguments")
    for k, v in vars(args).items():
        logger.debug(f"{k}: {v}")

    if args.task in ['ball', 'court']:
        if args.mode == 'train':
            train_model(args, logger)

        elif args.mode == 'val':
            validate_model(args, logger)

        elif args.mode == 'demo':
            demo_model(args, logger)

    elif args.task == 'bounce' and args.mode == 'train':
        train_bounce_model(args, logger)


if __name__ == '__main__':
    main()
