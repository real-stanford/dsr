import torch


# Rotation about the X-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.7)
def create_rotx(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 1, 1] = torch.cos(theta)
    rot[:, 2, 2] = rot[:, 1, 1]
    rot[:, 1, 2] = torch.sin(theta)
    rot[:, 2, 1] = -rot[:, 1, 2]
    return rot


# Rotation about the Y-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.6)
def create_roty(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 0, 0] = torch.cos(theta)
    rot[:, 2, 2] = rot[:, 0, 0]
    rot[:, 2, 0] = torch.sin(theta)
    rot[:, 0, 2] = -rot[:, 2, 0]
    return rot


# Rotation about the Z-axis by theta
# From Barfoot's book: http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser15.pdf (6.5)
def create_rotz(theta):
    N = theta.size(0)
    rot = torch.eye(3).type_as(theta).view(1, 3, 3).repeat(N, 1, 1)
    rot[:, 0, 0] = torch.cos(theta)
    rot[:, 1, 1] = rot[:, 0, 0]
    rot[:, 0, 1] = torch.sin(theta)
    rot[:, 1, 0] = -rot[:, 0, 1]
    return rot


# Create a skew-symmetric matrix "S" of size [B x 3 x 3] (passed in) given a [B x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    vec = vector.contiguous().view(N, 3)
    output = vec.new().resize_(N, 3, 3).fill_(0)
    output[:, 0, 1] = -vec[:, 2]
    output[:, 1, 0] = vec[:, 2]
    output[:, 0, 2] = vec[:, 1]
    output[:, 2, 0] = -vec[:, 1]
    output[:, 1, 2] = -vec[:, 0]
    output[:, 2, 1] = vec[:, 0]
    return output


# Compute the rotation matrix R from a set of unit-quaternions (N x 4):
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 9)
def create_rot_from_unitquat(unitquat):
    # Init memory
    N = unitquat.size(0)
    rot = unitquat.new_zeros([N, 3, 3])

    # Get quaternion elements. Quat = [qx,qy,qz,qw] with the scalar at the rear
    x, y, z, w = unitquat[:, 0], unitquat[:, 1], unitquat[:, 2], unitquat[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w

    # Row 1
    rot[:, 0, 0] = w2 + x2 - y2 - z2  # rot(0,0) = w^2 + x^2 - y^2 - z^2
    rot[:, 0, 1] = 2 * (x * y - w * z)  # rot(0,1) = 2*x*y - 2*w*z
    rot[:, 0, 2] = 2 * (x * z + w * y)  # rot(0,2) = 2*x*z + 2*w*y

    # Row 2
    rot[:, 1, 0] = 2 * (x * y + w * z)  # rot(1,0) = 2*x*y + 2*w*z
    rot[:, 1, 1] = w2 - x2 + y2 - z2  # rot(1,1) = w^2 - x^2 + y^2 - z^2
    rot[:, 1, 2] = 2 * (y * z - w * x)  # rot(1,2) = 2*y*z - 2*w*x

    # Row 3
    rot[:, 2, 0] = 2 * (x * z - w * y)  # rot(2,0) = 2*x*z - 2*w*y
    rot[:, 2, 1] = 2 * (y * z + w * x)  # rot(2,1) = 2*y*z + 2*w*x
    rot[:, 2, 2] = w2 - x2 - y2 + z2  # rot(2,2) = w^2 - x^2 - y^2 + z^2

    return rot


# Compute the derivatives of the rotation matrix w.r.t the unit quaternion
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 33-36)
def compute_grad_rot_wrt_unitquat(unitquat):
    # Compute dR/dq' (9x4 matrix)
    N = unitquat.size(0)
    x, y, z, w = unitquat.narrow(1, 0, 1), unitquat.narrow(1, 1, 1), unitquat.narrow(1, 2, 1), unitquat.narrow(1, 3, 1)
    dRdqh_w = 2 * torch.cat([w, -z, y, z, w, -x, -y, x, w], 1).view(N, 9, 1)  # Eqn 33, rows first
    dRdqh_x = 2 * torch.cat([x, y, z, y, -x, -w, z, w, -x], 1).view(N, 9, 1)  # Eqn 34, rows first
    dRdqh_y = 2 * torch.cat([-y, x, w, x, y, z, -w, z, -y], 1).view(N, 9, 1)  # Eqn 35, rows first
    dRdqh_z = 2 * torch.cat([-z, -w, x, w, -z, y, x, y, z], 1).view(N, 9, 1)  # Eqn 36, rows first
    dRdqh = torch.cat([dRdqh_x, dRdqh_y, dRdqh_z, dRdqh_w], 2)  # N x 9 x 4

    return dRdqh


# Compute the derivatives of a unit quaternion w.r.t a quaternion
def compute_grad_unitquat_wrt_quat(unitquat, quat):
    # Compute the quaternion norms
    N = quat.size(0)
    unitquat_v = unitquat.view(-1, 4, 1)
    norm2 = (quat * quat).sum(1)  # Norm-squared
    norm = torch.sqrt(norm2)  # Length of the quaternion

    # Compute gradient dq'/dq
    # TODO: No check for normalization issues currently
    I = torch.eye(4).view(1, 4, 4).expand(N, 4, 4).type_as(quat)
    qQ = torch.bmm(unitquat_v, unitquat_v.transpose(1, 2))  # q'*q'^T
    dqhdq = (I - qQ) / (norm.view(N, 1, 1).expand_as(I))

    return dqhdq


# Compute the derivatives of a unit quaternion w.r.t a SP quaternion
# From: http://www.tech.plymouth.ac.uk/sme/springerusv/2011/publications_files/Terzakis%20et%20al%202012,%20A%20Recipe%20on%20the%20Parameterization%20of%20Rotation%20Matrices...MIDAS.SME.2012.TR.004.pdf (Eqn 42-45)
def compute_grad_unitquat_wrt_spquat(spquat):
    # Compute scalars
    N = spquat.size(0)
    x, y, z = spquat.narrow(1, 0, 1), spquat.narrow(1, 1, 1), spquat.narrow(1, 2, 1)
    x2, y2, z2 = x * x, y * y, z * z
    s = 1 + x2 + y2 + z2  # 1 + x^2 + y^2 + z^2 = 1 + alpha^2
    s2 = (s * s).expand(N, 4)  # (1 + alpha^2)^2

    # Compute gradient dq'/dspq
    dqhdspq_x = (torch.cat([2 * s - 4 * x2, -4 * x * y, -4 * x * z, -4 * x], 1) / s2).view(N, 4, 1)
    dqhdspq_y = (torch.cat([-4 * x * y, 2 * s - 4 * y2, -4 * y * z, -4 * y], 1) / s2).view(N, 4, 1)
    dqhdspq_z = (torch.cat([-4 * x * z, -4 * y * z, 2 * s - 4 * z2, -4 * z], 1) / s2).view(N, 4, 1)
    dqhdspq = torch.cat([dqhdspq_x, dqhdspq_y, dqhdspq_z], 2)

    return dqhdspq


# Compute Unit Quaternion from SP-Quaternion
def create_unitquat_from_spquat(spquat):
    N = spquat.size(0)
    unitquat = spquat.new_zeros([N, 4])
    x, y, z = spquat[:, 0], spquat[:, 1], spquat[:, 2]
    alpha2 = x * x + y * y + z * z  # x^2 + y^2 + z^2
    unitquat[:, 0] = (2 * x) / (1 + alpha2)  # qx
    unitquat[:, 1] = (2 * y) / (1 + alpha2)  # qy
    unitquat[:, 2] = (2 * z) / (1 + alpha2)  # qz
    unitquat[:, 3] = (1 - alpha2) / (1 + alpha2)  # qw

    return unitquat
