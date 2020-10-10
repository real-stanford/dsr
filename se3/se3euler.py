import torch
from torch.autograd import Function
from se3.se3_utils import create_rotx, create_roty, create_rotz
from se3.se3_utils import create_skew_symmetric_matrix


class Se3euler(Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, num_se3, num_params = input.size()

        rot_params = input.view(batch_size * num_se3, -1)

        # Create rotations about X,Y,Z axes
        # R = Rz(theta3) * Ry(theta2) * Rx(theta1)
        # Last 3 parameters are [theta1, theta2 ,theta3]
        rotx = create_rotx(rot_params[:, 0])  # Rx(theta1)
        roty = create_roty(rot_params[:, 1])  # Ry(theta2)
        rotz = create_rotz(rot_params[:, 2])  # Rz(theta3)

        # Compute Rz(theta3) * Ry(theta2)
        rotzy = torch.bmm(rotz, roty)  # Rzy = R32

        # Compute rotation matrix R3*R2*R1 = R32*R1
        # R = Rz(t3) * Ry(t2) * Rx(t1)
        output = torch.bmm(rotzy, rotx)  # R = Rzyx

        ctx.save_for_backward(input, output, rotx, roty, rotz, rotzy)

        return output.view(batch_size, num_se3, 3, 3)

    @staticmethod
    def backward(ctx, grad_output):
        input, output, rotx, roty, rotz, rotzy = ctx.saved_tensors
        batch_size, num_se3, num_params = input.size()
        grad_output = grad_output.contiguous().view(batch_size * num_se3, 3, 3)

        # Gradient w.r.t Euler angles from Barfoot's book (http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf)
        grad_input_list = []
        for k in range(3):
            gradr = grad_output[:, k]  # Gradient w.r.t angle (k)
            vec = torch.zeros(1, 3).type_as(gradr)
            vec[0][k] = 1  # Unit vector
            skewsym = create_skew_symmetric_matrix(vec).view(1, 3, 3).expand_as(output)  # Skew symmetric matrix of unit vector
            if (k == 0):
                Rv = torch.bmm(torch.bmm(rotzy, skewsym), rotx)  # Eqn 6.61c
            elif (k == 1):
                Rv = torch.bmm(torch.bmm(rotz, skewsym), torch.bmm(roty, rotx))  # Eqn 6.61b
            else:
                Rv = torch.bmm(skewsym, output)
            grad_input_list.append(torch.sum(-Rv * grad_output, dim=(1, 2)))
        grad_input = torch.stack(grad_input_list, 1).view(batch_size, num_se3, 3)

        return grad_input
