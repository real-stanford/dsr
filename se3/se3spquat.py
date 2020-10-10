import torch
from torch.autograd import Function
from se3.se3_utils import create_unitquat_from_spquat
from se3.se3_utils import create_rot_from_unitquat
from se3.se3_utils import compute_grad_rot_wrt_unitquat
from se3.se3_utils import compute_grad_unitquat_wrt_spquat


class Se3spquat(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = create_unitquat_from_spquat(rot_params)

        output = create_rot_from_unitquat(unitquat).view(batch_size, num_se3, 3, 3)

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        unitquat = create_unitquat_from_spquat(rot_params)

        # Compute dR/dq'
        dRdqh = compute_grad_rot_wrt_unitquat(unitquat)

        # Compute dq'/dq = d(q/||q||)/dq = 1/||q|| (I - q'q'^T)
        dqhdspq = compute_grad_unitquat_wrt_spquat(rot_params)


        # Compute dR/dq = dR/dq' * dq'/dq
        dRdq = torch.bmm(dRdqh, dqhdspq).view(batch_size, num_se3, 3, 3, 3)  # B x k x 3 x 3 x 3

        # Scale by grad w.r.t output and sum to get gradient w.r.t quaternion params
        grad_out = grad_output.contiguous().view(batch_size, num_se3, 3, 3, 1).expand_as(dRdq)  # B x k x 3 x 3 x 3

        grad_input = torch.sum(dRdq * grad_out, dim=(2, 4))  # (Bk) x 3

        return grad_input
