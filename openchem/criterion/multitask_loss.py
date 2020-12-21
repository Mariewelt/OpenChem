import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss  #, _assert_no_grad


class MultitaskLoss(_WeightedLoss):
    r"""
    Creates a criterion that calculated binary cross-entropy loss over
    `n_tasks` tasks given input tensors `input` and `target`. Returns loss
    averaged across number of samples in every task and across `n_tasks`.
    It is useful when training a classification model with `n_tasks`
    separate binary classes.

    The loss can be described as:

    ..math::
        \text{loss}(y, t) = -\frac{1}{n_tasks}\sum_{i=1}^{n_tasks}\frac{1}{N_i}
        \sum_{j=1}^{N_i} \left(t[i, j]\log(1-y[i, j])
        + (1-t[i, j])\log(1-y[i, j])\right).

    Args:
        ignore_index (int): specifies a target value that is ignored
            and does not contribute to the gradient. For every task losses are
            averaged only across non-ignored targets.
        n_tasks (int): specifies number of tasks.

    Shape:
        -Input: :math: `(N, n_tasks)`. Values should be in :math:`[0, 1]` range,
            corresponding to probability of class :math:'1'.
        -Target: :math: `(N, n_tasks)`. Values should be binary: either
            :math:`0` or :math:`1`, corresponding to class labels.
        -Output: scalar.

    """
    def __init__(self, ignore_index, n_tasks):
        super(MultitaskLoss, self).__init__(reduction='none')
        self.n_tasks = n_tasks
        self.ignore_index = ignore_index

    def forward(self, input, target):
        assert target.size()[1] == self.n_tasks
        assert input.size()[1] == self.n_tasks
        x = torch.zeros(target.size()).cuda()
        y = torch.ones(target.size()).cuda()
        mask = torch.where(target == self.ignore_index, x, y)
        loss = F.binary_cross_entropy(input, mask * target, weight=self.weight)
        loss = loss * mask
        n_samples = mask.sum(dim=0)
        return (loss.sum(dim=0) / n_samples).mean()
