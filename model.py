from torch import nn
import torch
import torch.nn.functional as F
from model_utils import ConvBlock3D, ResBlock3D, ConvBlock2D, MLP
from se3.se3_module import SE3
from forward_warp import Forward_Warp_Cupy


class VolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 12
        self.conv00 = ConvBlock3D(input_channel, 16, stride=2, dilation=1, norm=True, relu=True) # 64x64x24

        self.conv10 = ConvBlock3D(16, 32, stride=2, dilation=1, norm=True, relu=True) # 32x32x12
        self.conv11 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv12 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv13 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)

        self.conv20 = ConvBlock3D(32, 64, stride=2, dilation=1, norm=True, relu=True) # 16x16x6
        self.conv21 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv22 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv23 = ConvBlock3D(64, 64, stride=1, dilation=1, norm=True, relu=True)

        self.conv30 = ConvBlock3D(64, 128, stride=2, dilation=1, norm=True, relu=True) # 8x8x3
        self.resn31 = ResBlock3D(128, 128)
        self.resn32 = ResBlock3D(128, 128)


    def forward(self, x):
        x0 = self.conv00(x)

        x1 = self.conv10(x0)
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x1 = self.conv13(x1)

        x2 = self.conv20(x1)
        x2 = self.conv21(x2)
        x2 = self.conv22(x2)
        x2 = self.conv23(x2)

        x3 = self.conv30(x2)
        x3 = self.resn31(x3)
        x3 = self.resn32(x3)

        return x3, (x2, x1, x0)


class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = ConvBlock3D(128, 64, norm=True, relu=True, upsm=True) # 16x16x6
        self.conv01 = ConvBlock3D(64, 64, norm=True, relu=True)

        self.conv10 = ConvBlock3D(64 + 64, 32, norm=True, relu=True, upsm=True) # 32x32x12
        self.conv11 = ConvBlock3D(32, 32, norm=True, relu=True)

        self.conv20 = ConvBlock3D(32 + 32, 16, norm=True, relu=True, upsm=True) # 64X64X24
        self.conv21 = ConvBlock3D(16, 16, norm=True, relu=True)

        self.conv30 = ConvBlock3D(16 + 16, 8, norm=True, relu=True, upsm=True) # 128X128X48
        self.conv31 = ConvBlock3D(8, 8, norm=True, relu=True)

    def forward(self, x, cache):
        m0, m1, m2 = cache

        x0 = self.conv00(x)
        x0 = self.conv01(x0)

        x1 = self.conv10(torch.cat([x0, m0], dim=1))
        x1 = self.conv11(x1)

        x2 = self.conv20(torch.cat([x1, m1], dim=1))
        x2 = self.conv21(x2)

        x3 = self.conv30(torch.cat([x2, m2], dim=1))
        x3 = self.conv31(x3)

        return x3


class MotionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d00 = ConvBlock3D(8 + 8, 8, stride=2, dilation=1, norm=True, relu=True)  # 64

        self.conv3d10 = ConvBlock3D(8 + 8, 16, stride=2, dilation=1, norm=True, relu=True)  # 32

        self.conv3d20 = ConvBlock3D(16 + 16, 32, stride=2, dilation=1, norm=True, relu=True)  # 16

        self.conv3d30 = ConvBlock3D(32, 16, dilation=1, norm=True, relu=True, upsm=True) # 32
        self.conv3d40 = ConvBlock3D(16, 8, dilation=1, norm=True, relu=True, upsm=True) # 64
        self.conv3d50 = ConvBlock3D(8, 8, dilation=1, norm=True, relu=True, upsm=True) # 128
        self.conv3d60 = nn.Conv3d(8, 3, kernel_size=3, padding=1)


        self.conv2d10 = ConvBlock2D(8, 64, stride=2, norm=True, relu=True)  # 64
        self.conv2d11 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d12 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d13 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d14 = ConvBlock2D(64, 8, stride=1, dilation=1, norm=True, relu=True)

        self.conv2d20 = ConvBlock2D(64, 128, stride=2, norm=True, relu=True)  # 32
        self.conv2d21 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d22 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d23 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d24 = ConvBlock2D(128, 16, stride=1, dilation=1, norm=True, relu=True)

    def forward(self, feature, action):
        # feature: [B, 8, 128, 128, 48]
        # action:  [B, 8,  128, 128]

        action_embedding0 = torch.unsqueeze(action, -1).expand([-1, -1, -1, -1, 48])
        feature0 = self.conv3d00(torch.cat([feature, action_embedding0], dim=1))

        action1 = self.conv2d10(action)
        action1 = self.conv2d11(action1)
        action1 = self.conv2d12(action1)
        action1 = self.conv2d13(action1)

        action_embedding1 = self.conv2d14(action1)
        action_embedding1 = torch.unsqueeze(action_embedding1, -1).expand([-1, -1, -1, -1, 24])
        feature1 = self.conv3d10(torch.cat([feature0, action_embedding1], dim=1))

        action2 = self.conv2d20(action1)
        action2 = self.conv2d21(action2)
        action2 = self.conv2d22(action2)
        action2 = self.conv2d23(action2)

        action_embedding2 = self.conv2d24(action2)
        action_embedding2 = torch.unsqueeze(action_embedding2, -1).expand([-1, -1, -1, -1, 12])
        feature2 = self.conv3d20(torch.cat([feature1, action_embedding2], dim=1))

        feature3 = self.conv3d30(feature2)
        feature4 = self.conv3d40(feature3 + feature1)
        feature5 = self.conv3d50(feature4 + feature0)

        motion_pred = self.conv3d60(feature5)

        return motion_pred

class MaskDecoder(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.decoder = nn.Conv3d(8, K, kernel_size=1)

    def forward(self, x):
        logit = self.decoder(x)
        mask = torch.softmax(logit, dim=1)
        return logit, mask


class TransformDecoder(nn.Module):
    def __init__(self, transform_type, object_num):
        super().__init__()
        num_params_dict = {
            'affine': 12,
            'se3euler': 6,
            'se3aa': 6,
            'se3spquat': 6,
            'se3quat': 7
        }
        self.num_params = num_params_dict[transform_type]
        self.object_num = object_num

        self.conv3d00 = ConvBlock3D(8 + 8, 8, stride=2, dilation=1, norm=True, relu=True)  # 64

        self.conv3d10 = ConvBlock3D(8 + 8, 16, stride=2, dilation=1, norm=True, relu=True)  # 32

        self.conv3d20 = ConvBlock3D(16 + 16, 32, stride=2, dilation=1, norm=True, relu=True)  # 16
        self.conv3d21 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv3d22 = ConvBlock3D(32, 32, stride=1, dilation=1, norm=True, relu=True)
        self.conv3d23 = ConvBlock3D(32, 64, stride=1, dilation=1, norm=True, relu=True)

        self.conv3d30 = ConvBlock3D(64, 128, stride=2, dilation=1, norm=True, relu=True)  # 8

        self.conv3d40 = ConvBlock3D(128, 128, stride=2, dilation=1, norm=True, relu=True)  # 4

        self.conv3d50 = nn.Conv3d(128, 128, kernel_size=(4, 4, 2))


        self.conv2d10 = ConvBlock2D(8, 64, stride=2, norm=True, relu=True)  # 64
        self.conv2d11 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d12 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d13 = ConvBlock2D(64, 64, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d14 = ConvBlock2D(64, 8, stride=1, dilation=1, norm=True, relu=True)

        self.conv2d20 = ConvBlock2D(64, 128, stride=2, norm=True, relu=True)  # 32
        self.conv2d21 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d22 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d23 = ConvBlock2D(128, 128, stride=1, dilation=1, norm=True, relu=True)
        self.conv2d24 = ConvBlock2D(128, 16, stride=1, dilation=1, norm=True, relu=True)

        self.mlp = MLP(
            input_dim=128,
            output_dim=self.num_params * self.object_num,
            hidden_sizes=[512, 512, 512, 512],
            hidden_nonlinearity=F.leaky_relu
        )


    def forward(self, feature, action):
        # feature: [B, 8, 128, 128, 48]
        # action:  [B, 8,  128, 128]

        action_embedding0 = torch.unsqueeze(action, -1).expand([-1, -1, -1, -1, 48])
        feature0 = self.conv3d00(torch.cat([feature, action_embedding0], dim=1))

        action1 = self.conv2d10(action)
        action1 = self.conv2d11(action1)
        action1 = self.conv2d12(action1)
        action1 = self.conv2d13(action1)

        action_embedding1 = self.conv2d14(action1)
        action_embedding1 = torch.unsqueeze(action_embedding1, -1).expand([-1, -1, -1, -1, 24])
        feature1 = self.conv3d10(torch.cat([feature0, action_embedding1], dim=1))

        action2 = self.conv2d20(action1)
        action2 = self.conv2d21(action2)
        action2 = self.conv2d22(action2)
        action2 = self.conv2d23(action2)

        action_embedding2 = self.conv2d24(action2)
        action_embedding2 = torch.unsqueeze(action_embedding2, -1).expand([-1, -1, -1, -1, 12])
        feature2 = self.conv3d20(torch.cat([feature1, action_embedding2], dim=1))
        feature2 = self.conv3d21(feature2)
        feature2 = self.conv3d22(feature2)
        feature2 = self.conv3d23(feature2)

        feature3 = self.conv3d30(feature2)
        feature4 = self.conv3d40(feature3)
        feature5 = self.conv3d50(feature4)

        params = self.mlp(feature5.view([-1, 128]))
        params = params.view([-1, self.object_num, self.num_params])

        return params


class ModelDSR(nn.Module):
    def __init__(self, object_num=5, transform_type='se3euler', motion_type='se3'):
        # transform_type options:   None, 'affine', 'se3euler', 'se3aa', 'se3quat', 'se3spquat'
        # motion_type options:      'se3', 'conv'
        # input volume size:        [128, 128, 48]

        super().__init__()
        self.transform_type = transform_type
        self.K = object_num
        self.motion_type = motion_type

        # modules
        self.forward_warp = Forward_Warp_Cupy.apply
        self.volume_encoder = VolumeEncoder()
        self.feature_decoder = FeatureDecoder()
        if self.motion_type == 'se3':
            self.mask_decoder = MaskDecoder(self.K)
            self.transform_decoder = TransformDecoder(
                transform_type=self.transform_type,
                object_num=self.K - 1
            )
            self.se3 = SE3(self.transform_type)
        elif self.motion_type == 'conv':
            self.motion_decoder = MotionDecoder()
        else:
            raise ValueError('motion_type doesn\'t support ', self.motion_type)

        # initialization
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv3d) or isinstance(m[1], nn.Conv2d):
                nn.init.kaiming_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm3d) or isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

        # const value
        self.grids = torch.stack(torch.meshgrid(
            torch.linspace(0, 127, 128),
            torch.linspace(0, 127, 128),
            torch.linspace(0, 47, 48)
        ))
        self.coord_feature = self.grids / torch.tensor([128, 128, 48]).view([3, 1, 1, 1])
        self.grids_flat = self.grids.view(1, 1, 3, 128 * 128 * 48)
        self.zero_vec = torch.zeros([1, 1, 3], dtype=torch.float)
        self.eye_mat = torch.eye(3, dtype=torch.float)

    def forward(self, input_volume, last_s=None, input_action=None, input_motion=None, next_mask=False, no_warp=False):
        B, _, S1, S2, S3 = input_volume.size()
        K = self.K
        device = input_volume.device
        output = {}

        input = torch.cat((input_volume, self.coord_feature.expand(B, -1, -1, -1, -1).to(device)), dim=1)
        input = torch.cat((input, last_s), dim=1) # aggregate history

        volume_embedding, cache = self.volume_encoder(input)
        mask_feature = self.feature_decoder(volume_embedding, cache)

        if self.motion_type == 'conv':
            motion = self.motion_decoder(mask_feature, input_action)
            output['motion'] = motion

            return output


        assert(self.motion_type == 'se3')
        logit, mask = self.mask_decoder(mask_feature)
        output['init_logit'] = logit
        transform_param = self.transform_decoder(mask_feature, input_action)

        # trans, pivot: [B, K-1, 3]
        # rot_matrix:   [B, K-1, 3, 3]
        trans_vec, rot_mat = self.se3(transform_param)
        mask_object = torch.narrow(mask, 1, 0, K - 1)
        sum_mask = torch.sum(mask_object, dim=(2, 3, 4))
        heatmap = torch.unsqueeze(mask_object, dim=2) * self.grids.to(device)
        pivot_vec = torch.sum(heatmap, dim=(3, 4, 5)) / torch.unsqueeze(sum_mask, dim=2)

        # [Important] The last one is the background!
        trans_vec = torch.cat([trans_vec, self.zero_vec.expand(B, -1, -1).to(device)], dim=1).unsqueeze(-1)
        rot_mat = torch.cat([rot_mat, self.eye_mat.expand(B, 1, -1, -1).to(device)], dim=1)
        pivot_vec = torch.cat([pivot_vec, self.zero_vec.expand(B, -1, -1).to(device)], dim=1).unsqueeze(-1)

        grids_flat = self.grids_flat.to(device)
        grids_after_flat = rot_mat @ (grids_flat - pivot_vec) + pivot_vec + trans_vec
        motion = (grids_after_flat - grids_flat).view([B, K, 3, S1, S2, S3])

        motion = torch.sum(motion * torch.unsqueeze(mask, 2), 1)

        output['motion'] = motion

        if no_warp:
            output['s'] = mask_feature
        elif input_motion is not None:
            mask_feature_warp = self.forward_warp(
                mask_feature,
                input_motion,
                torch.sum(mask[:, :-1, ], dim=1)
            )
            output['s'] = mask_feature_warp
        else:
            mask_feature_warp = self.forward_warp(
                mask_feature,
                motion,
                torch.sum(mask[:, :-1, ], dim=1)
            )
            output['s'] = mask_feature_warp

        if next_mask:
            mask_warp = self.forward_warp(
                mask,
                motion,
                torch.sum(mask[:, :-1, ], dim=1)
            )
            output['next_mask'] = mask_warp

        return output



    def get_init_repr(self, batch_size):
        return torch.zeros([batch_size, 8, 128, 128, 48], dtype=torch.float)


if __name__=='__main__':
    torch.cuda.set_device(4)
    model = ModelDSR(
        object_num=2,
        transform_type='se3euler',
        with_history=True,
        motion_type='se3'
    ).cuda()

    input_volume = torch.rand((4, 1, 128, 128, 48)).cuda()
    input_action = torch.rand((4, 8, 128, 128)).cuda()
    last_s = model.get_init_repr(4).cuda()

    output = model(input_volume=input_volume, last_s=last_s, input_action=input_action, next_mask=True)

    for k in output.keys():
        print(k, output[k].size())


