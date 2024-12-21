import torch


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


@torch.no_grad()
def accuracy(output, target):
    predicted = torch.argmax(output, dim=1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    accuracy = 100.0 * correct / total

    return accuracy

