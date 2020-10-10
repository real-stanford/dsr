import numpy as np
import torch
from torch.autograd import Function
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple


class Forward_Warp_Cupy(Function):
    @staticmethod
    def forward(ctx, feature, flow, mask):
        kernel = '''
        extern "C"
        __global__ void warp_forward(
            const float * im0, // [B, C, W, H, D]
            const float * flow, // [B, 3, W, H, D]
            const float * mask, // [B, W, H, D]
            float * im1, // [B, C, W, H, D]
            float * cnt, // [B, W, H, D]
            const int vol_batch,
            const int vol_dim_x,
            const int vol_dim_y,
            const int vol_dim_z,
            const int feature_dim,
            const int warp_mode //0 (bilinear), 1 (nearest)
        ) {
            // Get voxel index
            int max_threads_per_block = blockDim.x;
            int block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
            int voxel_idx = block_idx * max_threads_per_block + threadIdx.x;
            
            int voxel_size_product = vol_dim_x * vol_dim_y * vol_dim_z;
            
            // IMPORTANT
            if (voxel_idx >= vol_batch * voxel_size_product) return;
            
            // Get voxel grid coordinates (note: be careful when casting)
            int tmp = voxel_idx;
            
            int voxel_z = tmp % vol_dim_z;
            tmp = tmp / vol_dim_z;
            
            int voxel_y = tmp % vol_dim_y;
            tmp = tmp / vol_dim_y;
            
            int voxel_x = tmp % vol_dim_x;
            int batch = tmp / vol_dim_x;
            
            int voxel_idx_BCWHD = voxel_idx + batch * (voxel_size_product * (feature_dim - 1));
            int voxel_idx_flow = voxel_idx + batch * (voxel_size_product * (3 - 1));
            
            // Main part
            if (warp_mode == 0) {
                // bilinear
                float x_float = voxel_x + flow[voxel_idx_flow];
                float y_float = voxel_y + flow[voxel_idx_flow + voxel_size_product];
                float z_float = voxel_z + flow[voxel_idx_flow + voxel_size_product + voxel_size_product];
                
                int x_floor = x_float;
                int y_floor = y_float;
                int z_floor = z_float;
                
                for(int t = 0; t < 8; t++) {
                    int dx = (t >= 4);
                    int dy = (t - 4 * dx) >= 2;
                    int dz = t - 4 * dx - dy * 2;
                    
                    int x = x_floor + dx;
                    int y = y_floor + dy;
                    int z = z_floor + dz;
                    
                    if (x >= 0 && x < vol_dim_x && y >= 0 && y < vol_dim_y && z >= 0 && z < vol_dim_z) {
                        float weight = mask[voxel_idx];
                        weight *= (dx == 0 ? (x_floor + 1 - x_float) : (x_float - x_floor));
                        weight *= (dy == 0 ? (y_floor + 1 - y_float) : (y_float - y_floor));
                        weight *= (dz == 0 ? (z_floor + 1 - z_float) : (z_float - z_floor));
                        int idx = (((int)batch * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                        atomicAdd(&cnt[idx], weight);
                        
                        int idx_BCWHD = (((int)batch * feature_dim * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                        
                        for(int c = 0, offset = 0; c < feature_dim; c++, offset += voxel_size_product) {
                            atomicAdd(&im1[idx_BCWHD + offset], im0[voxel_idx_BCWHD + offset] * weight);
                        }
                    }
                    
                }
            } else {
                // nearest
                int x = round(voxel_x + flow[voxel_idx_flow]);
                int y = round(voxel_y + flow[voxel_idx_flow + voxel_size_product]);
                int z = round(voxel_z + flow[voxel_idx_flow + voxel_size_product + voxel_size_product]);
                
                if (x >= 0 && x < vol_dim_x && y >= 0 && y < vol_dim_y && z >= 0 && z < vol_dim_z) {
                    int idx = (((int)batch * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                    float mask_weight = mask[voxel_idx];
                    atomicAdd(&cnt[idx], mask_weight);
                    
                    int idx_BCWHD = (((int)batch * feature_dim * vol_dim_x + x) * vol_dim_y + y) * vol_dim_z + z;
                    
                    for(int c = 0, offset = 0; c < feature_dim; c++, offset += voxel_size_product) {
                        atomicAdd(&im1[idx_BCWHD + offset], im0[voxel_idx_BCWHD + offset] * mask_weight);
                    }
                }
            }
        }
        '''
        program = Program(kernel, 'warp_forward.cu')
        ptx = program.compile()
        m = function.Module()
        m.load(bytes(ptx.encode()))
        f = m.get_function('warp_forward')
        Stream = namedtuple('Stream', ['ptr'])
        s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        B, C, W, H, D = feature.size()
        warp_mode = 0
        n_blocks = np.ceil(B * W * H * D / 1024.0)
        grid_dim_x = int(np.cbrt(n_blocks))
        grid_dim_y = int(np.sqrt(n_blocks / grid_dim_x))
        grid_dim_z = int(np.ceil(n_blocks / grid_dim_x / grid_dim_y))
        assert grid_dim_x * grid_dim_y * grid_dim_z * 1024 >= B * W * H * D

        feature_new = torch.zeros_like(feature)
        cnt = torch.zeros_like(mask)

        f(grid=(grid_dim_x, grid_dim_y, grid_dim_z), block=(1024, 1, 1),
          args=[feature.data_ptr(), flow.data_ptr(), mask.data_ptr(),  feature_new.data_ptr(), cnt.data_ptr(),
                B, W, H, D, C, warp_mode], stream=s)

        eps=1e-3
        cnt = torch.max(cnt, other=torch.ones_like(cnt) * eps)
        feature_new = feature_new / torch.unsqueeze(cnt, 1)

        return feature_new

    @staticmethod
    def backward(ctx, feature_new_grad):
        # Not implemented
        return None, None, None