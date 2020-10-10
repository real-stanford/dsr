import os.path as osp
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import itertools
import shutil
from tqdm import tqdm
from data import Data
from utils import mkdir, flow2im, html_visualize, mask_visualization, tsdf_visualization
from model import ModelDSR

parser = argparse.ArgumentParser()

# exp args
parser.add_argument('--exp', type=str, help='name of exp')
parser.add_argument('--gpus', type=int, nargs='+', help='list of gpus to be used, separated by space')
parser.add_argument('--resume', default=None, type=str, help='path to model or exp, None means training from scratch')

# data args
parser.add_argument('--data_path', type=str, help='path to data')
parser.add_argument('--object_num', type=int, default=5, help='number of objects')
parser.add_argument('--seq_len', type=int, default=10, help='sequence length for training')
parser.add_argument('--batch', type=int, default=12, help='batch size per gpu')
parser.add_argument('--workers', type=int, default=4, help='number of workers per gpu')

parser.add_argument('--model_type', type=str, default='dsr', choices=['dsr', 'single', 'nowarp', 'gtwarp', '3dflow'])
parser.add_argument('--transform_type', type=str, default='se3euler', choices=['affine', 'se3euler', 'se3aa', 'se3spquat', 'se3quat'])

# loss args
parser.add_argument('--alpha_motion', type=float, default=1.0, help='weight of motino loss (MSE)')
parser.add_argument('--alpha_mask', type=float, default=5.0, help='weight of mask loss (BCE)')

# training args
parser.add_argument('--snapshot_freq', type=int, default=1, help='snapshot frequency')
parser.add_argument('--epoch', type=int, default=30, help='number of training eposhes')
parser.add_argument('--finetune', dest='finetune', action='store_true',
    help='finetuning or training from scratch ==> different learning rate strategies')

# distributed training args
parser.add_argument('--seed', type=int, default=23333, help='random seed')
parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed training backend')
parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:2333', help='distributed training url')


def main():
    args = parser.parse_args()

    # loss types & loss_idx
    loss_types = ['all', 'motion', 'mask']
    loss_idx = {}
    for i, loss_type in enumerate(loss_types):
        loss_idx[loss_type] = i
    print('==> loss types: ', loss_types)
    args.loss_types = loss_types
    args.loss_idx = loss_idx

    # check sequence length
    if args.model_type == 'single':
        assert(args.seq_len == 1)

    # resume
    if args.resume is not None and not args.resume.endswith('.pth'):
        args.resume = osp.join('exp', args.resume, 'models/latest.pth')

    # dir & args
    exp_dir = osp.join('exp', args.exp)
    mkdir(exp_dir)

    print('==> arguments parsed')
    str_list = []
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))
        str_list.append('--{0}={1} \\'.format(key, getattr(args, key)))

    args.model_dir = osp.join(exp_dir, 'models')
    mkdir(args.model_dir)
    args.visualization_dir = osp.join(exp_dir, 'visualization')
    mkdir(args.visualization_dir)

    mp.spawn(main_worker, nprocs=len(args.gpus), args=(len(args.gpus), args))


def main_worker(rank, world_size, args):
    args.gpu = args.gpus[rank]
    if rank == 0:
        writer = SummaryWriter(osp.join('exp', args.exp))
    print(f'==> Rank={rank}, Use GPU: {args.gpu} for training.')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=rank)

    torch.cuda.set_device(args.gpu)

    model = ModelDSR(
        object_num=args.object_num,
        transform_type=args.transform_type,
        motion_type='se3' if args.model_type != '3dflow' else 'conv',
    )

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.95))

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=torch.device(f'cuda:{args.gpu}'))
        model.load_state_dict(checkpoint['state_dict'])
        print(f'==> rank={rank}, loaded checkpoint {args.resume}')

    data, samplers, loaders = {}, {}, {}
    for split in ['train', 'test']:
        data[split] = Data(data_path=args.data_path, split=split, seq_len=args.seq_len)
        samplers[split] = torch.utils.data.distributed.DistributedSampler(data[split])
        loaders[split] = DataLoader(
            dataset=data[split],
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=samplers[split],
            pin_memory=False
        )
    print('==> dataset loaded: [size] = {0} + {1}'.format(len(data['train']), len(data['test'])))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    for epoch in range(args.epoch):
        samplers['train'].set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        if rank == 0:
            print(f'==> epoch = {epoch}, lr = {lr}')

        with torch.enable_grad():
            loss_tensor_train = iterate(loaders['train'], model, optimizer, rank, args)
        with torch.no_grad():
            loss_tensor_test = iterate(loaders['test'], model, None, rank, args)

        # tensorboard log
        loss_tensor = torch.stack([loss_tensor_train, loss_tensor_test]).cuda()
        torch.distributed.all_reduce(loss_tensor)
        if rank == 0:
            training_step = (epoch + 1) * len(data['train'])
            loss_tensor = loss_tensor.cpu().numpy()
            for i, split in enumerate(['train', 'test']):
                for j, loss_type in enumerate(args.loss_types):
                    for step_id in range(args.seq_len):
                        writer.add_scalar(
                            '%s-loss_%s/%d' % (split, loss_type, step_id),
                            loss_tensor[i, j, step_id] / len(data[split]), epoch+1)
            writer.add_scalar('learning_rate', lr, epoch + 1)

        if rank == 0 and (epoch + 1) % args.snapshot_freq == 0:
            visualize(loaders, model, epoch, args)
            save_state = {
                'state_dict': model.module.state_dict(),
            }
            torch.save(save_state, osp.join(args.model_dir, 'latest.pth'))
            shutil.copyfile(
                osp.join(args.model_dir, 'latest.pth'),
                osp.join(args.model_dir, 'epoch_%d.pth' % (epoch + 1))
            )

def adjust_learning_rate(optimizer, epoch, args):
    if args.finetune:
        if epoch < 5:
            lr = 5e-4
        elif epoch < 10:
            lr = 2e-4
        elif epoch < 15:
            lr = 5e-5
        else:
            lr = 1e-5

    else:
        if epoch < 2:
            lr = 1e-5
        elif epoch < 5:
            lr = 1e-3
        elif epoch < 10:
            lr = 5e-4
        elif epoch < 20:
            lr = 2e-4
        elif epoch < 25:
            lr = 5e-5
        else:
            lr = 1e-5
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def iterate(loader, model, optimizer, rank, args):
    motion_metric = nn.MSELoss()
    loss_tensor = torch.zeros([len(args.loss_types), args.seq_len])
    if rank == 0:
        loader = tqdm(loader, desc='test' if optimizer is None else 'train')
    for batch in loader:
        batch_size = batch['0-action'].size(0)
        last_s = model.module.get_init_repr(batch_size).cuda()
        batch_order = None

        for step_id in range(args.seq_len):
            output = model(
                input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                last_s=last_s,
                input_action=batch['%d-action' % step_id].cuda(),
                input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                no_warp=args.model_type=='nowarp'
            )
            last_s = output['s'].data
            loss = 0

            if 'motion' in args.loss_types:
                loss_motion = motion_metric(
                    output['motion'],
                    batch['%d-scene_flow_3d' % step_id].cuda()
                )
                loss_tensor[args.loss_idx['motion'], step_id] += loss_motion.item() * batch_size
                loss += args.alpha_motion * loss_motion

            if 'mask' in args.loss_types:
                mask_gt = batch['%d-mask_3d' % step_id].cuda()
                if batch_order is None:
                    batch_order = get_batch_order(output['init_logit'], mask_gt)
                loss_mask = get_mask_loss(output['init_logit'], mask_gt, batch_order)
                loss_tensor[args.loss_idx['mask'], step_id] += loss_mask.item() * batch_size
                loss += args.alpha_mask * loss_mask

            loss_tensor[args.loss_idx['all'],  step_id] += loss.item() * batch_size

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_id != args.seq_len - 1:
                batch_order = get_batch_order(output['init_logit'], mask_gt)
    return loss_tensor


def get_batch_order(logit_pred, mask_gt):
    batch_order = []
    B, K, S1, S2, S3 = logit_pred.size()
    sum = 0
    for b in range(B):
        all_p = list(itertools.permutations(list(range(K - 1))))
        best_loss, best_p = None, None
        for p in all_p:
            permute_pred = torch.stack(
                [logit_pred[b:b + 1, -1]] + [logit_pred[b:b + 1, i] for i in p],
                dim=1).contiguous()
            cur_loss = nn.CrossEntropyLoss()(permute_pred, mask_gt[b:b + 1]).item()
            if best_loss is None or cur_loss < best_loss:
                best_loss = cur_loss
                best_p = p
        batch_order.append(best_p)
        sum += best_loss
    return batch_order


def get_mask_loss(logit_pred, mask_gt, batch_order):
    loss = 0
    B, K, S1, S2, S3 = logit_pred.size()
    for b in range(B):
        permute_pred = torch.stack(
            [logit_pred[b:b + 1, -1]] + [logit_pred[b:b + 1, i] for i in batch_order[b]],
            dim=1).contiguous()
        loss += nn.CrossEntropyLoss()(permute_pred, mask_gt[b:b + 1])
    return loss


def visualize(loaders, model, epoch, args):
    visualization_path = osp.join(args.visualization_dir, 'epoch_%03d' % (epoch + 1))
    figures = {}
    ids = [split + '_' + str(itr) + '-' + str(step_id)
           for split in ['train', 'test']
           for itr in range(args.batch)
           for step_id in range(args.seq_len)]
    cols = ['color_image', 'color_heightmap', 'motion_gt', 'motion_pred', 'mask_gt']
    if args.model_type != '3dflow':
        cols = cols + ['mask_pred', 'next_mask_pred']

    with torch.no_grad():
        for split in ['train', 'test']:
            model.train()
            batch = iter(loaders[split]).next()
            batch_size = batch['0-action'].size(0)
            last_s = model.module.get_init_repr(batch_size).cuda()
            for step_id in range(args.seq_len):
                output = model(
                    input_volume=batch['%d-tsdf' % step_id].cuda().unsqueeze(1),
                    last_s=last_s,
                    input_action=batch['%d-action' % step_id].cuda(),
                    input_motion=batch['%d-scene_flow_3d' % step_id].cuda() if args.model_type=='gtwarp' else None,
                    no_warp=args.model_type=='nowarp',
                    next_mask=True
                )
                last_s = output['s'].data

                vis_color_image = batch['%d-color_image' % step_id].numpy()
                vis_color_heightmap = batch['%d-color_heightmap' % step_id].numpy()
                motion_gt = torch.sum(batch['%d-scene_flow_3d' % step_id][:, :2, ...], dim=4).numpy()
                motion_pred = torch.sum(output['motion'][:, :2, ...], dim=4).cpu().numpy()

                vis_mask_gt = mask_visualization(batch['%d-mask_3d' % step_id].numpy())

                if args.model_type != '3dflow':
                    vis_mask_pred = mask_visualization(output['init_logit'].cpu().numpy())
                    vis_next_mask_pred = mask_visualization(output['next_mask'].cpu().numpy())

                for k in range(args.batch):
                    figures['%s_%d-%d_color_image' % (split, k, step_id)] = vis_color_image[k]
                    figures['%s_%d-%d_color_heightmap' % (split, k, step_id)] = vis_color_heightmap[k]
                    figures['%s_%d-%d_motion_gt' % (split, k, step_id)] = flow2im(motion_gt[k])
                    figures['%s_%d-%d_motion_pred' % (split, k, step_id)] = flow2im(motion_pred[k])
                    figures['%s_%d-%d_mask_gt' % (split, k, step_id)] = vis_mask_gt[k]
                    if args.model_type != '3dflow':
                        figures['%s_%d-%d_mask_pred' % (split, k, step_id)] = vis_mask_pred[k]
                        figures['%s_%d-%d_next_mask_pred' % (split, k, step_id)] = vis_next_mask_pred[k]

    html_visualize(visualization_path, figures, ids, cols, title=args.exp)


if __name__ == '__main__':
    main()