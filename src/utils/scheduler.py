from torch.optim.lr_scheduler import StepLR


def select_scheduler(name, optimizer, args):
    if name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args['epochs'] // 10, gamma=0.5)
    else:
        raise Exception('Unknown scheduler')
    return scheduler
