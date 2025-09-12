from torch import nn

def select_loss(name):
    if name == 'MSE':
        return nn.MSELoss()
    elif name == 'MAE':
        return nn.L1Loss()
    elif name == 'BCE':
        return nn.BCELoss()
    elif name == 'CE':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss')
