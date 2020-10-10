import torch
from torch.autograd import Function
import torch.nn.functional as F
from se3.se3_utils import create_skew_symmetric_matrix


class Se3aa(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()
        N = batch_size * num_se3
        eps = 1e-12

        rot_params = input.view(batch_size * num_se3, -1)

        # Get the un-normalized axis and angle
        axis = rot_params.view(N, 3, 1)  # Un-normalized axis
        angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
        angle = torch.sqrt(angle2)  # Angle

        # Compute skew-symmetric matrix "K" from the axis of rotation
        K = create_skew_symmetric_matrix(axis)
        K2 = torch.bmm(K, K)  # K * K

        # Compute sines
        S = torch.sin(angle) / angle
        S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

        # Compute cosines
        C = (1 - torch.cos(angle)) / angle2
        C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

        # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
        rot = torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(rot_params)  # R = I
        rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
        rot += K2 * C.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2

        ctx.save_for_backward(input, rot)

        return rot.view(batch_size, num_se3, 3, 3)

    @staticmethod
    def backward(ctx, grad_output):
        input, rot = ctx.saved_tensors
        batch_size, num_se3, num_params = input.size()
        N = batch_size * num_se3
        eps = 1e-12
        grad_output =grad_output.contiguous().view(N, 3, 3)

        rot_params = input.view(batch_size * num_se3, -1)

        axis = rot_params.view(N, 3, 1)  # Un-normalized axis
        angle2 = (axis * axis).sum(1)  # (Bk) x 1 x 1 => Norm of the vector (squared angle)
        nSmall = angle2.lt(eps).sum()  # Num angles less than threshold

        # Compute: v x (Id - R) for all the columns of (Id-R)
        I = torch.eye(3).type_as(input).repeat(N, 1, 1).add(-1, rot)  # (Bk) x 3 x 3 => Id - R
        vI = torch.cross(axis.expand_as(I), I, 1)  # (Bk) x 3 x 3 => v x (Id - R)

        # Compute [v * v' + v x (Id - R)] / ||v||^2
        vV = torch.bmm(axis, axis.transpose(1, 2))  # (Bk) x 3 x 3 => v * v'
        vV = (vV + vI) / (angle2.view(N, 1, 1).expand_as(vV))  # (Bk) x 3 x 3 => [v * v' + v x (Id - R)] / ||v||^2

        # Iterate over the 3-axis angle parameters to compute their gradients
        # ([v * v' + v x (Id - R)] / ||v||^2 _ k) x (R) .* gradOutput  where "x" is the cross product
        grad_input_list = []
        for k in range(3):
            # Create skew symmetric matrix
            skewsym = create_skew_symmetric_matrix(vV.narrow(2, k, 1))

            # For those AAs with angle^2 < threshold, gradient is different
            # We assume angle = 0 for these AAs and update the skew-symmetric matrix to be one w.r.t identity
            if (nSmall > 0):
                vec = torch.zeros(1, 3).type_as(skewsym)
                vec[0][k] = 1  # Unit vector
                idskewsym = create_skew_symmetric_matrix(vec)
                for i in range(N):
                    if (angle2[i].squeeze()[0] < eps):
                        skewsym[i].copy_(idskewsym.squeeze())  # Use the new skew sym matrix (around identity)

            # Compute the gradients now
            grad_input_list.append(torch.sum(torch.bmm(skewsym, rot) * grad_output, dim=(1, 2)))  # [(Bk) x 1 x 1] => (vV x R) .* gradOutput
        grad_input = torch.stack(grad_input_list, 1).view(batch_size, num_se3, 3)

        return grad_input
