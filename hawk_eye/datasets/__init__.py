def get_dataloader(args):
    if args.task == 'ball':
        if args.num_seq > 1:
            from hawk_eye.datasets.ball_seq import get_ballseq_dataloader
            return get_ballseq_dataloader(args)
        else:
            from hawk_eye.datasets.ball import get_ball_dataloader
            return get_ball_dataloader(args)

    elif args.task == 'court':
        from hawk_eye.datasets.court import get_court_dataloader
        return get_court_dataloader(args)

    else:
        from hawk_eye.datasets.bounce import get_bounce_datasets
        return get_bounce_datasets(args)
