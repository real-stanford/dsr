from torch.nn.modules import Module
from se3.se3spquat import Se3spquat
from se3.se3quat import Se3quat
from se3.se3euler import Se3euler
from se3.se3aa import Se3aa

class SE3(Module):
    def __init__(self, transform_type='affine', has_pivot=False):
        super().__init__()
        rot_param_num_dict = {
            'affine': 9,
            'se3euler': 3,
            'se3aa': 3,
            'se3spquat': 3,
            'se3quat': 4
        }
        self.transform_type = transform_type
        self.rot_param_num = rot_param_num_dict[transform_type]
        self.has_pivot = has_pivot
        self.num_param = rot_param_num_dict[transform_type] + 3
        if self.has_pivot:
            self.num_param += 3

    def forward(self, input):
        B, K, L = input.size()
        if L != self.num_param:
            raise ValueError('Dimension Error!')

        trans_vec = input.narrow(2, 0, 3)
        rot_params = input.narrow(2, 3, self.rot_param_num)
        if self.has_pivot:
            pivot_vec = input.narrow(2, 3 + self.rot_param_num, 3)


        if self.transform_type == 'affine':
            rot_mat = rot_params.view(B, K, 3, 3)
        elif self.transform_type == 'se3euler':
            rot_mat = Se3euler.apply(rot_params)
        elif self.transform_type == 'se3aa':
            rot_mat = Se3aa.apply(rot_params)
        elif self.transform_type == 'se3spquat':
            rot_mat = Se3spquat.apply(rot_params)
        elif self.transform_type == 'se3quat':
            rot_mat = Se3quat.apply(rot_params)

        if self.has_pivot:
            return trans_vec, rot_mat, pivot_vec
        else:
            return trans_vec, rot_mat