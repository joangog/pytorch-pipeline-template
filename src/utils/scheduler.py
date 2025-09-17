from torch.optim.lr_scheduler import StepLR


def get_scheduler(name, optimizer, args):
    if name == 'StepLR':
        n_steps = 10
        assert args['epochs'] >= n_steps, \
            f'For StepLR scheduler, epochs should be at least {n_steps} because the step size is {n_steps}'
        scheduler = StepLR(optimizer, step_size=args['epochs'] // n_steps,
                           gamma=0.5)  # LR will be halved every n_steps (one step is one epoch)
    elif name is None:
        return None
    else:
        raise ValueError('Unknown scheduler')
    return scheduler
