import os.path as osp
import collections
import math
import os
import shutil

import cv2
import imageio
import numpy as np
import dominate
from dominate.tags import *
import queue
import threading


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def transform_points(pts, transform):
    # pts = [3xN] array
    # transform: [3x4]
    pts_t = np.dot(transform[0:3, 0:3], pts) + np.tile(transform[0:3, 3:], (1, pts.shape[1]))
    return pts_t


def project_pts_to_2d(pts, camera_view_matrix, camera_intrisic):
    # transformation from word to virtual camera
    # camera_intrisic for virtual camera [ [f,0,0],[0,f,0],[0,0,1]] f is focal length
    # RT_wrd2cam
    pts_c = transform_points(pts, camera_view_matrix[0:3, :])
    rot_algix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    pts_c = transform_points(pts_c, rot_algix)
    coord_2d = np.dot(camera_intrisic, pts_c)
    coord_2d[0:2, :] = coord_2d[0:2, :] / np.tile(coord_2d[2, :], (2, 1))
    coord_2d[2, :] = pts_c[2, :]
    return coord_2d


def project_pts_to_3d(color_image, depth_image, camera_intr, camera_pose):
    W, H = depth_image.shape
    cam_pts, rgb_pts = get_pointcloud(color_image, depth_image, camera_intr)
    world_pts = np.transpose(
        np.dot(camera_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(camera_pose[0:3, 3:], (1, cam_pts.shape[0])))

    pts = world_pts.reshape([W, H, 3])
    pts = np.transpose(pts, [2, 0, 1])

    return pts


def get_pointcloud(color_img, depth_img, camera_intrinsics):
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    cam_pts_x = np.multiply(pix_x - camera_intrinsics[0, 2], depth_img / camera_intrinsics[0, 0])
    cam_pts_y = np.multiply(pix_y - camera_intrinsics[1, 2], depth_img / camera_intrinsics[1, 1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h * im_w, 1)
    cam_pts_y.shape = (im_h * im_w, 1)
    cam_pts_z.shape = (im_h * im_w, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r.shape = (im_h * im_w, 1)
    rgb_pts_g.shape = (im_h * im_w, 1)
    rgb_pts_b.shape = (im_h * im_w, 1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
                               (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) + np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(
        np.logical_and(surface_pts[:, 0] >= workspace_limits[0][0], surface_pts[:, 0] < workspace_limits[0][1]),
        surface_pts[:, 1] >= workspace_limits[1][0]), surface_pts[:, 1] < workspace_limits[1][1]),
        surface_pts[:, 2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    # depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap


def mkdir(path, clean=False):
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def imresize(im, dsize, cfirst=False):
    if cfirst:
        im = im.transpose(1, 2, 0)
    im = cv2.resize(im, dsize=dsize)
    if cfirst:
        im = im.transpose(2, 0, 1)
    return im


def imretype(im, dtype):
    im = np.array(im)

    if im.dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(np.float)
    elif im.dtype == 'uint8':
        im = im.astype(np.float) / 255.
    elif im.dtype == 'uint16':
        im = im.astype(np.float) / 65535.
    else:
        raise NotImplementedError('unsupported source dtype: {0}'.format(im.dtype))

    assert np.min(im) >= 0 and np.max(im) <= 1

    if dtype in ['float', 'float16', 'float32', 'float64']:
        im = im.astype(dtype)
    elif dtype == 'uint8':
        im = (im * 255.).astype(dtype)
    elif dtype == 'uint16':
        im = (im * 65535.).astype(dtype)
    else:
        raise NotImplementedError('unsupported target dtype: {0}'.format(dtype))

    return im


def imwrite(path, obj):
    if not isinstance(obj, (collections.Sequence, collections.UserList)):
        obj = [obj]
    writer = imageio.get_writer(path)
    for im in obj:
        im = imretype(im, dtype='uint8').squeeze()
        if len(im.shape) == 3 and im.shape[0] == 3:
            im = np.transpose(im, (1, 2, 0))
        writer.append_data(im)
    writer.close()


def flow2im(flow, max=None, dtype='float32', cfirst=False):
    flow = np.array(flow)

    if np.ndim(flow) == 3 and flow.shape[0] == 2:
        x, y = flow[:, ...]
    elif np.ndim(flow) == 3 and flow.shape[-1] == 2:
        x = flow[..., 0]
        y = flow[..., 1]
    else:
        raise NotImplementedError(
            'unsupported flow size: {0}'.format(flow.shape))

    rho, theta = cv2.cartToPolar(x, y)

    if max is None:
        max = np.maximum(np.max(rho), 1e-6)

    hsv = np.zeros(list(rho.shape) + [3], dtype=np.uint8)
    hsv[..., 0] = theta * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(rho / max, 1) * 255

    im = cv2.cvtColor(hsv, code=cv2.COLOR_HSV2RGB)
    im = imretype(im, dtype=dtype)

    if cfirst:
        im = im.transpose(2, 0, 1)
    return im


def draw_arrow(image, action, direction_num=8, heightmap_pixel_size=0.004):
    # image: [W, H, 3] (color image) or [W, H] (depth image)
    def put_in_bound(val, bound):
        # output: 0 <= val < bound
        val = min(max(0, val), bound - 1)
        return val

    img = image.copy()
    if isinstance(action, tuple):
        x_ini, y_ini, direction = action
    else:
        x_ini, y_ini, direction = action['2'], action['1'], action['0']

    pushing_distance = 0.15

    angle = direction / direction_num * 2 * np.pi
    x_end = put_in_bound(int(x_ini + pushing_distance / heightmap_pixel_size * np.cos(angle)), image.shape[1])
    y_end = put_in_bound(int(y_ini + pushing_distance / heightmap_pixel_size * np.sin(angle)), image.shape[0])

    if img.shape[0] == 1:
        # gray img, white arrow
        img = imretype(img[:, :, np.newaxis], 'uint8')
        cv2.arrowedLine(img=img, pt1=(x_ini, y_ini), pt2=(x_end, y_end), color=255, thickness=2, tipLength=0.2)
    elif img.shape[2] == 3:
        # rgb img, red arrow
        cv2.arrowedLine(img=img, pt1=(x_ini, y_ini), pt2=(x_end, y_end), color=(255, 0, 0), thickness=2, tipLength=0.2)
    return img


def multithreading_exec(num, q, fun, blocking=True):
    """
    Multi-threading Execution

    :param num: number of threadings
    :param q: queue of args
    :param fun: function to be executed
    :param blocking: blocking or not (default True)
    """

    class Worker(threading.Thread):
        def __init__(self, q, fun):
            super().__init__()
            self.q = q
            self.fun = fun
            self.start()

        def run(self):
            while True:
                try:
                    args = self.q.get(block=False)
                    self.fun(*args)
                    self.q.task_done()
                except queue.Empty:
                    break

    thread_list = [Worker(q, fun) for i in range(num)]
    if blocking:
        for t in thread_list:
            if t.is_alive():
                t.join()


def html_visualize(web_path, data, ids, cols, others=[], title='visualization', threading_num=10):
    """
    :param web_path: (str) directory to save webpage. It will clear the old data!
    :param data: (dict of data).
        key: {id}_{col}.
        value: figure or text
            - figure: ndarray --> .png or [ndarrays,] --> .gif
            - text: str or [str,]
    :param ids: (list of str) name of each row
    :param cols: (list of str) name of each column
    :param others: (list of dict) other figures
        'name': str, name of the data, visualize using h2()
        'data': string or ndarray(image)
        'height': int, height of the image (default 256)
    :param title: (str) title of the webpage
    :param threading_num: number of threadings for imwrite (default 10)
    """
    figure_path = os.path.join(web_path, 'figures')
    mkdir(web_path, clean=True)
    mkdir(figure_path, clean=True)
    q = queue.Queue()
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            q.put((os.path.join(figure_path, key + '.png'), value))
        if not isinstance(value, list) and isinstance(value[0], np.ndarray):
            q.put((os.path.join(figure_path, key + '.gif'), value))
    multithreading_exec(threading_num, q, imwrite)

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style='table-layout: fixed;'):
            with dominate.tags.tr():
                with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    dominate.tags.p('id')
                for col in cols:
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', ):
                        dominate.tags.p(col)
            for id in ids:
                with dominate.tags.tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center',
                                          bgcolor=bgcolor):
                        for part in id.split('_'):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                            value = data.get(f'{id}_{col}', None)
                            if isinstance(value, str):
                                dominate.tags.p(value)
                            elif isinstance(value, list) and isinstance(value[0], str):
                                for v in value:
                                    dominate.tags.p(v)
                            else:
                                dominate.tags.img(style='height:128px',
                                                  src=os.path.join('figures', '{}_{}.png'.format(id, col)))
        for idx, other in enumerate(others):
            dominate.tags.h2(other['name'])
            if isinstance(other['data'], str):
                dominate.tags.p(other['data'])
            else:
                imwrite(os.path.join(figure_path, '_{}_{}.png'.format(idx, other['name'])), other['data'])
                dominate.tags.img(style='height:{}px'.format(other.get('height', 256)),
                                  src=os.path.join('figures', '_{}_{}.png'.format(idx, other['name'])))
    with open(os.path.join(web_path, 'index.html'), 'w') as fp:
        fp.write(web.render())


def mask_visualization(mask):
    # mask: numpy array, [B, K, W, H, D] or [B, W, H, D]
    # Red, Green, Blue, Yellow, Purple
    colors = [(255, 87, 89), (89, 169, 79), (78, 121, 167), (237, 201, 72), (176, 122, 161)]
    if len(mask.shape) == 5:
        B, K, W, H, D = mask.shape
        argmax_mask = np.argmax(mask, axis=1)
    else:
        B, W, H, D = mask.shape
        K = max(np.max(mask) + 1, 2)
        argmax_mask = mask - 1

    mask_list = []
    for k in range(K - 1):
        mask_list.append((argmax_mask == k).astype(np.float32))

    mask = np.sum(np.stack(mask_list, axis=1), axis=4)
    sum_mask = np.sum(mask, 1) + 1  # [B, 1, W, H]
    color_mask = np.zeros([B, W, H, 3])
    for i in range(K - 1):
        color_mask += mask[:, i, ..., np.newaxis] * np.array(colors[i])
    return np.clip(color_mask / sum_mask[..., np.newaxis] / 255.0, 0, 1)


def mask_visualization_2d(mask):
    # Red, Green, Blue, Yellow, Purple
    colors = [(255, 87, 89), (89, 169, 79), (78, 121, 167), (237, 201, 72), (176, 122, 161)]
    if len(mask.shape) == 4:
        B, K, W, H = mask.shape
        argmax_mask = np.argmax(mask, axis=1)
    else:
        B, W, H = mask.shape
        K = max(np.max(mask) + 1, 2)
        argmax_mask = mask
    mask_list = []
    for k in range(K):
        mask_list.append((argmax_mask == k).astype(np.float32))

    mask = np.stack(mask_list, axis=1)
    sum_mask = np.sum(mask, 1) + 1  # [B, 1, W, H]
    color_mask = np.zeros([B, W, H, 3])
    for i in range(K):
        color_mask += mask[:, i, ..., np.newaxis] * np.array(colors[i])
    return np.clip(color_mask / sum_mask[..., np.newaxis] / 255.0, 0, 1)


def volume_visualization(volume):
    # volume: numpy array: [B, W, H, D]
    tmp = np.sum(volume, axis=-1)
    tmp -= np.min(tmp)
    tmp /= max(np.max(tmp), 1)
    return np.clip(tmp, 0, 1)


def tsdf_visualization(tsdf):
    # tsdf: numpy array: [B, W, H, D]
    return volume_visualization((tsdf < 0).astype(np.float32))
