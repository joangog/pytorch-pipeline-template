from torch.optim import SGD


def select_optimizer(name, params, args):
    if name == 'SGD':
        optimizer = SGD(params, lr=args['learning_rate'], momentum=args['momentum'],
                        weight_decay=args['weight_decay'])
    else:
        raise ValueError('Unknown optimizer')
    return optimizer
