import torch
import torch.nn as nn
import torch.nn.functional as F


class StackAugmentation(nn.Module):
    def __init__(self, stack_width, stack_depth, in_features, use_cuda):
        super(StackAugmentation, self).__init__()
        self.use_cuda = use_cuda
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.in_features = in_features
        self.stack_controls_layer = nn.Linear(in_features=in_features, out_features=3)

        self.stack_input_layer = nn.Linear(in_features=in_features, out_features=self.stack_width)

    def forward(self, input_val, prev_stack):
        batch_size = prev_stack.size(0)

        controls = self.stack_controls_layer(input_val.squeeze(0))
        controls = F.softmax(controls, dim=1)
        controls = controls.view(-1, 3, 1, 1)
        stack_input = self.stack_input_layer(input_val)
        stack_input = F.tanh(stack_input)
        stack_input = stack_input.permute(1, 0, 2)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = torch.tensor(zeros_at_the_bottom.cuda(), requires_grad=True)
        else:
            zeros_at_the_bottom = torch.tensor(zeros_at_the_bottom, requires_grad=True)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((stack_input, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + \
                    a_pop * stack_down
        return new_stack

    def init_stack(self, batch_size):
        result = torch.zeros(batch_size, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return torch.tensor(result.cuda(), requires_grad=True)
        else:
            return torch.tensor(result, requires_grad=True)
