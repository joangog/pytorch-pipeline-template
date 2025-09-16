from torch.optim.lr_scheduler import StepLR


def get_scheduler(name, optimizer, args):
    if name == 'StepLR':
        # TODO: Fix scheduler, it doesnt work and make the loss never go down (maybe fix the step or sth)
        scheduler = StepLR(optimizer, step_size=args['epochs'] // 10, gamma=0.5)
    elif name is None:
        return None
    else:
        raise ValueError('Unknown scheduler')
    return scheduler
