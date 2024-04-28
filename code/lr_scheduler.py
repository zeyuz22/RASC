from torch.optim import lr_scheduler


def get_inv_lr_scheduler(optimizer, gamma=0.0001, power=0.75, last_epoch=-1):
    lr_lambda = lambda epoch: (1 + gamma * epoch) ** (-power)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
