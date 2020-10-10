import torch
from torch.autograd import Function
import torch.nn.functional as F
from se3.se3_utils import create_rot_from_unitquat
from se3.se3_utils import compute_grad_rot_wrt_unitquat
from se3.se3_utils import compute_grad_unitquat_wrt_quat


class Se3quat(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = F.normalize(rot_params)

        output = create_rot_from_unitquat(unitquat).view(batch_size, num_se3, 3, 3)

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = F.normalize(rot_params)

        # Compute dR/dq'
        dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

        # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
        dqhdq = compute_grad_unitquat_wrt_quat(unitquat, rot_params)


        # Compute dR/dq = dR/dq' * dq'/dq
        dRdq = torch.bmm(dRdqh, dqhdq).view(batch_size, num_se3, 3, 3, 4)  # B x k x 3 x 3 x 4

        # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
        grad_out = grad_output.contiguous().view(batch_size, num_se3, 3, 3, 1).expand_as(dRdq)  # B x k x 3 x 3 x 4

        grad_input = torch.sum(dRdq * grad_out, dim=(2, 3))  # (Bk) x 3

        return grad_input
