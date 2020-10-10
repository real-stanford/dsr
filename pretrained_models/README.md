# Pretrained Models

### The following pretrained models are provided:
- [dsr](dsr.pth): DSR-Net introduced in the paper. (without real data finetuning)
- [dsr_ft](dsr_ft.pth): DSR-Net introduced in the paper. (with real data finetuning)
- [single](single.pth): It does not use any history aggregation.
- [nowarp](nowarp.pth): It does not warp the representation before aggregation.
- [gtwarp](gtwarp.pth): It warps the representation with ground truth motion (i.e., performance oracle)
- [3dflow](3dflow.pth): It predicts per-voxel scene flow for the entire 3D volume.
