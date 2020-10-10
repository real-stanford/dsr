import numpy as np
import torch
import argparse
from tqdm import tqdm
import os.path as osp
from data import Data
from torch.utils.data import DataLoader
from model import ModelDSR
import itertools


parser = argparse.ArgumentParser()

parser.add_argument('--resume', type=str, help='path to model')
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--test_type', type=str, choices=['motion_visible', 'motion_full', 'mask_ordered', 'mask_unordered'])

parser.add_argument('--gpu', type=int, default=0, help='gpu id (single gpu)')
parser.add_argument('--object_num', type=int, default=5, help='number of objects')
parser.add_argument('--seq_len', type=int, default=10, help='sequence length')
parser.add_argument('--batch', type=int, default=12, help='batch size')
parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')

parser.add_argument('--model_type', type=str, default='dsr', choices=['dsr', 'single', 'nowarp', 'gtwarp', '3dflow'])
parser.add_argument('--transform_type', type=str, default='se3euler', choices=['affine', 'se3euler', 'se3aa', 'se3spquat', 'se3quat'])

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    data, loaders = {}, {}
    for split in ['test']:
        data[split] = Data(data_path=args.data_path, split=split, seq_len=args.seq_len)
        loaders[split] = DataLoader(dataset=data[split], batch_size=args.batch, num_workers=args.workers)
    print('==> dataset loaded: [size] = {0}'.format(len(data['test'])))


    model = ModelDSR(
        object_num=args.object_num,
        transform_type=args.transform_type,
        motion_type='se3' if args.model_type != '3dflow' else 'conv',
    )
    model.cuda()

    checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{args.gpu}'))
    model.load_state_dict(checkpoint['state_dict'])
    print('==> resume: ' + args.resume)

    with torch.no_grad():
        if args.test_type == 'motion_visible':
            evaluation_motion_visible(args, model, loaders['test'])
        
        if args.test_type == 'motion_full':
            evaluation_motion_full(args, model, loaders['test'])

        if args.test_type == 'mask_ordered':
            evaluation_mask_ordered(args, model, loaders['test'])

        if args.test_type == 'mask_unordered':
            evaluation_mask_unordered(args, model, loaders['test'])

def evaluation_mask_unordered(args, model, loader):
    print(f'==> evaluation_mask (unordered)')
    iou_dict = [[] for _ in range(args.seq_len)]
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        last_s = model.get_init_repr(batch_size).cuda()
        logit_pred_list, mask_gt_list = [], []
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type == 'single':
                last_s = output['s'].data

            logit_pred = output['init_logit']
            mask_gt = batch['%d-mask_3d' % step_id].cuda()
            iou_unordered = calc_iou_unordered(logit_pred, mask_gt)
            iou_dict[step_id].append(iou_unordered)
    print('mask_unordered (IoU) = ', np.mean([np.mean(np.concatenate(iou_dict[i])) for i in range(args.seq_len)]))


def calc_iou_unordered(logit_pred, mask_gt_argmax):
    # logit_pred: [B, K, S1, S2, S3], softmax, the last channel is empty
    # mask_gt_argmax: [B, S1, S2, S3], 0 represents empty
    B, K, S1, S2, S3 = logit_pred.size()
    logit_pred_argmax = torch.argmax(logit_pred, dim=1, keepdim=True)
    mask_gt_argmax = torch.unsqueeze(mask_gt_argmax, 1)
    mask_pred_onehot = torch.zeros_like(logit_pred).scatter(1, logit_pred_argmax, 1)[:, :-1]
    mask_gt_onehot = torch.zeros_like(logit_pred).scatter(1, mask_gt_argmax, 1)[:, 1:]
    K -= 1
    info_dict = {'I': np.zeros([B, K, K]), 'U': np.zeros([B, K, K])}
    for b in range(B):
        for i in range(K):
            for j in range(K):
                mask_gt = mask_gt_onehot[b, i]
                mask_pred = mask_pred_onehot[b, j]
                I = torch.sum(mask_gt * mask_pred).item()
                U = torch.sum(mask_gt + mask_pred).item() - I
                info_dict['I'][b, i, j] = I
                info_dict['U'][b, i, j] = U
    batch_ious = []
    for b in range(B):
        best_iou, best_p = 0, None
        for p in list(itertools.permutations(range(K))):
            cur_I = [info_dict['I'][b, i, p[i]] for i in range(K)]
            cur_U = [info_dict['U'][b, i, p[i]] for i in range(K)]
            cur_iou = np.mean(np.array(cur_I) / np.maximum(np.array(cur_U), 1))
            if cur_iou > best_iou:
                best_iou = cur_iou
        batch_ious.append(best_iou)

    return np.array(batch_ious)


def evaluation_mask_ordered(args, model, loader):
    print(f'==> evaluation_mask (ordered)')
    iou_dict = []
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        last_s = model.get_init_repr(batch_size).cuda()
        logit_pred_list, mask_gt_list = [], []
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type == 'single':
                last_s = output['s'].data

            logit_pred = output['init_logit']
            mask_gt = batch['%d-mask_3d' % step_id].cuda()
            logit_pred_list.append(logit_pred)
            mask_gt_list.append(mask_gt)
        iou_ordered = calc_iou_ordered(logit_pred_list, mask_gt_list)
        iou_dict.append(iou_ordered)
    print('mask_ordered (IoU) = ', np.mean(np.concatenate(iou_dict)))


def calc_iou_ordered(logit_pred_list, mask_gt_argmax_list):
    # logit_pred_list: [L, B, K, S1, S2, S3], softmax, the last channel is empty
    # mask_gt_argmax_list: [L, B, S1, S2, S3], 0 represents empty
    L = len(logit_pred_list)
    B, K, S1, S2, S3 = logit_pred_list[0].size()
    K -= 1
    info_dict = {'I': np.zeros([L, B, K, K]), 'U': np.zeros([L, B, K, K])}
    for l in range(L):
        logit_pred = logit_pred_list[l]
        mask_gt_argmax = mask_gt_argmax_list[l]
        logit_pred_argmax = torch.argmax(logit_pred, dim=1, keepdim=True)
        mask_gt_argmax = torch.unsqueeze(mask_gt_argmax, 1)
        mask_pred_onehot = torch.zeros_like(logit_pred).scatter(1, logit_pred_argmax, 1)[:, :-1]
        mask_gt_onehot = torch.zeros_like(logit_pred).scatter(1, mask_gt_argmax, 1)[:, 1:]
        for b in range(B):
            for i in range(K):
                for j in range(K):
                    mask_gt = mask_gt_onehot[b, i]
                    mask_pred = mask_pred_onehot[b, j]
                    I = torch.sum(mask_gt * mask_pred).item()
                    U = torch.sum(mask_gt + mask_pred).item() - I
                    info_dict['I'][l, b, i, j] = I
                    info_dict['U'][l, b, i, j] = U
    batch_ious = []
    for b in range(B):
        best_iou, best_p = 0, None
        for p in list(itertools.permutations(range(K))):
            cur_I = [info_dict['I'][l, b, i, p[i]] for l in range(L) for i in range(K)]
            cur_U = [info_dict['U'][l, b, i, p[i]] for l in range(L) for i in range(K)]
            cur_iou = np.mean(np.array(cur_I) / np.maximum(np.array(cur_U), 1))
            if cur_iou > best_iou:
                best_iou = cur_iou
        batch_ious.append(best_iou)

    return np.array(batch_ious)


def evaluation_motion_visible(args, model, loader):
    print('==> evaluation_motion (visible surface)')
    mse_dict = [0 for _ in range(args.seq_len)]
    data_num = 0
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        data_num += batch_size
        last_s = model.get_init_repr(batch_size).cuda()
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type in ['single', '3dflow'] :
                last_s = output['s'].data

            tsdf = batch['%d-tsdf' % step_id].cuda().unsqueeze(1)
            mask = batch['%d-mask_3d' % step_id].cuda().unsqueeze(1)
            surface_mask = ((tsdf > -0.99).float()) * ((tsdf < 0).float()) * ((mask > 0).float())
            surface_mask[..., 0] = 0

            target = batch['%d-scene_flow_3d' % step_id].cuda()
            pred = output['motion']

            mse = torch.sum((target - pred) ** 2 * surface_mask, dim=[1, 2, 3, 4]) / torch.sum(surface_mask, dim=[1, 2, 3, 4])
            mse_dict[step_id] += torch.sum(mse).item() * 0.16
            # 0.16(0.4^2) is the scale to convert the unit from "voxel" to "cm".
            # The voxel size is 0.4cm. Here we use seuqre error.
    print('motion_visible (MSE in cm) = ', np.mean([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]))


def evaluation_motion_full(args, model, loader):
    print('==> evaluation_motion (full volume)')
    mse_dict = [0 for _ in range(args.seq_len)]
    data_num = 0
    for batch in tqdm(loader):
        batch_size = batch['0-action'].size(0)
        data_num += batch_size
        last_s = model.get_init_repr(batch_size).cuda()
        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            if not args.model_type in ['single', '3dflow'] :
                last_s = output['s'].data

            target = batch['%d-scene_flow_3d' % step_id].cuda()
            pred = output['motion']

            mse = torch.mean((target - pred) ** 2, dim=[1, 2, 3, 4])
            mse_dict[step_id] += torch.sum(mse).item() * 0.16
            # 0.16(0.4^2) is the scale to convert the unit from "voxel" to "cm".
            # The voxel size is 0.4cm. Here we use seuqre error.
    print('motion_full (MSE in cm) = ', np.mean([np.mean(mse_dict[i]) / data_num for i in range(args.seq_len)]))


if __name__ == '__main__':
    main()