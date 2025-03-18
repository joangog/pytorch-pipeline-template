import torch

def load_checkpoint(model, optimizer, scheduler, weight_path, resume=False):
    # TODO: if resume, then load epoch, model state, optimizer state, lr scheduler state from provided checkpoint
    # TODO: if not resume, but weights is provided, then simply use only the model state
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    start_epoch = 0
    if resume:  # If resuming training from last epoch, then load optimizer and scheduler states
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler, start_epoch

