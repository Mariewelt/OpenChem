import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad


class MultitaskLoss(_WeightedLoss):
    def __init__(self, ignore_index, n_tasks):
        super(MultitaskLoss, self).__init__(reduce=False)
        self.n_tasks = n_tasks
        self.ignore_index = ignore_index

    def forward(self, output, target):
        _assert_no_grad(target)
        assert target.size()[1] == self.n_tasks
        assert output.size()[1] == self.n_tasks
        x = torch.zeros(target.size()).cuda()
        y = torch.ones(target.size()).cuda()
        mask = torch.where(target == self.ignore_index, x, y)
        loss = F.binary_cross_entropy(output, mask*target,
                                      weight=self.weight,
                                      size_average=self.size_average,
                                      reduce=self.reduce)
        loss = loss*mask
        n_samples = mask.sum(dim=0)
        return (loss.sum(dim=0)/n_samples).mean()
