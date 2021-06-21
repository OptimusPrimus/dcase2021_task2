import torch

class Lambda(torch.nn.Module):

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MeanStdNormalization(torch.nn.Module):

    def __init__(self, mean, std, requires_grad=True):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.from_numpy(mean[..., None]).float(), requires_grad=requires_grad)
        self.std = torch.nn.Parameter(torch.from_numpy(std[..., None]).float(), requires_grad=requires_grad)

    def forward(self, x):
        return (x - self.mean) / self.std

