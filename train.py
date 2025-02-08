"""Train code."""

import torch
from torch.optim.lr_scheduler import MultiStepLR

from datasets import get_dataloader
from models import get_model
from utils.misc import AverageMeter


def train_step(model, dataloader, criterion, optimizer, device, logger):
    """Train one epoch."""
    global global_step
    model.train()

    print_interval = int(len(dataloader) * 0.1)
    loss_meter = AverageMeter()
    for i, (inputs, targets) in enumerate(dataloader):
        global_step += 1
        outputs = model(inputs['image'].to(device))
        loss = criterion(outputs, targets, device)

        loss_meter.update(loss.item(), n=len(targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % print_interval == 0:
            logger.info(f"Step: {i+1}/{len(dataloader)}, Loss: {loss_meter.avg:1.4f}")

    return loss_meter.avg


def train_model(args, logger):
    """Train a model."""
    global global_step
    global_step = 0
    device = torch.device(args.device)

    model, criterion, postprocessor = get_model(args, device)
    optimizer, scheduler = get_optimizer(model, args)
    dataloader = get_dataloader(args)
    
    logger.info("Start training...")
    for epoch in range(args.epoch):
        loss = train_step(model, dataloader, criterion, optimizer, device, logger)
        scheduler.step()

        logger.info(f"Epoch: {epoch + 1}, Loss: {loss:1.4f}")

        save_path = args.save_root / 'weights' / f'ckpt.pth'
        if (epoch + 1) % 5 == 0:
            save_path = args.save_root / 'weights' / f'{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
                       
    torch.save(model.state_dict(),
               args.save_root / 'weights' / f'ckpt.pth')

    return None

def get_optimizer(model, args):
    """Get optimizer."""
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_milestone is not None:
        scheduler = MultiStepLR(optimizer=optimizer,
                                milestones=args.lr_milestone,
                                gamma=args.lr_gamma)
    return optimizer, scheduler


def train_bounce_model(args, logger):
    import catboost as ctb
    from sklearn.metrics import confusion_matrix, accuracy_score

    grid = {'iterations': [150, 200, 250],
            'learning_rate': [0.03, 0.1],
            'depth': [2, 4, 6],
            'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    
    model, _, _ = get_model(args, None)
    (x_train, y_train), (x_test, y_test) = get_dataloader(args)
    train_dataset = ctb.Pool(x_train, y_train)

    model.grid_search(grid, train_dataset)

    pred_ctb = model.predict(x_test)
    y_pred_bin = (pred_ctb > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel()
    
    logger.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    acc = accuracy_score(y_test, y_pred_bin)
    logger.info(f"Accuracy: {acc:1.4f}")

    model.save_model(args.save_root / 'weights' / 'ckpt.cbm')
    return None