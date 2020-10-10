import os.path as osp
import time
import cv2
import numpy as np
import pybullet as p
import json

from sim import PybulletSim
from binvox_utils import read_as_coord_array
from utils import euler2rotm, project_pts_to_2d


class SimulationEnv():
    def __init__(self, gui_enabled):

        self.gui_enabled = gui_enabled
        self.sim = PybulletSim(gui_enabled=gui_enabled, tool='stick')
        self.heightmap_size = self.sim._heightmap_size
        self.heightmap_pixel_size = self.sim._heightmap_pixel_size
        self.view_bounds = self.sim._view_bounds
        self.direction_num = 8
        self.voxel_size = 0.004

        self.object_ids = []

        self.object_type = 'cube'  # choice: 'cube', 'shapenet', 'ycb'

        # process ycb
        self.ycb_path = 'object_models/ycb'
        self.ycb_info = json.load(open('assets/object_id/ycb_id.json', 'r'))

        # process shapenent
        self.shapenet_path = 'object_models/shapenet'
        self.shapenet_info = json.load(open('assets/object_id/shapenet_id.json', 'r'))

        self.voxel_coord = {}
        self.cnt_dict = {}
        self.init_position = {}
        self.last_direction = {}


    def _get_coord(self, obj_id, position, orientation, vol_bnds=None, voxel_size=None):
        # if vol_bnds is not None, return coord in voxel, else, return world coord
        coord = self.voxel_coord[obj_id]
        mat = euler2rotm(p.getEulerFromQuaternion(orientation))
        coord = (mat @ (coord.T)).T + np.asarray(position)
        if vol_bnds is not None:
            coord = np.round((coord - vol_bnds[:, 0]) / voxel_size).astype(np.int)
        return coord

    def _get_scene_flow_3d(self, old_po_ors):
        vol_bnds = self.view_bounds
        scene_flow = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds] + [3])
        mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

        cur_cnt = 0
        for obj_id, old_po_or in zip(self.object_ids, old_po_ors):
            position, orientation = p.getBasePositionAndOrientation(obj_id)
            new_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

            position, orientation = old_po_or
            old_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

            motion = new_coord - old_coord

            valid_idx = np.logical_and(
                np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < 128),
                np.logical_and(
                    np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < 128),
                    np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < 48)
                )
            )
            x = old_coord[valid_idx, 1]
            y = old_coord[valid_idx, 0]
            z = old_coord[valid_idx, 2]
            motion = motion[valid_idx]
            motion = np.stack([motion[:, 1], motion[:, 0], motion[:, 2]], axis=1)

            scene_flow[x, y, z] = motion

            # mask
            cur_cnt += 1
            mask[x, y, z] = cur_cnt

        return mask, scene_flow

    def _get_scene_flow_2d(self, old_po_or):
        old_coords_world = []
        new_coords_world = []
        point_id_list = []
        cur_cnt = 0
        for obj_id, po_or in zip(self.object_ids, old_po_or):
            position, orientation = po_or
            old_coord = self._get_coord(obj_id, position, orientation)
            old_coords_world.append(old_coord)

            position, orientation = p.getBasePositionAndOrientation(obj_id)
            new_coord = self._get_coord(obj_id, position, orientation)
            new_coords_world.append(new_coord)

            cur_cnt += 1
            point_id_list.append([cur_cnt for _ in range(old_coord.shape[0])])

        point_id = np.concatenate(point_id_list)
        old_coords_world = np.concatenate(old_coords_world)
        new_coords_world = np.concatenate(new_coords_world)
        camera_view_matrix = np.array(self.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.sim.camera_params[1]['camera_intr']
        image_size = self.sim.camera_params[1]['camera_image_size']
        old_coords_2d = project_pts_to_2d(old_coords_world.T, camera_view_matrix, camera_intr)
        y = np.round(old_coords_2d[0]).astype(np.int)
        x = np.round(old_coords_2d[1]).astype(np.int)
        depth = old_coords_2d[2]
        valid_idx = np.logical_and(
            np.logical_and(x >= 0, x < image_size[0]),
            np.logical_and(y >= 0, y < image_size[1])
        )
        x = x[valid_idx]
        y = y[valid_idx]
        depth = depth[valid_idx]
        point_id = point_id[valid_idx]
        motion = (new_coords_world - old_coords_world)[valid_idx]

        sort_id = np.argsort(-depth)
        x = x[sort_id]
        y = y[sort_id]
        point_id = point_id[sort_id]
        motion = motion[sort_id]
        motion = np.stack([motion[:, 1], motion[:, 0], motion[:, 2]], axis=1)

        scene_flow = np.zeros([image_size[0], image_size[1], 3])
        mask = np.zeros([image_size[0], image_size[1]])

        scene_flow[x, y] = motion
        mask[x, y] = point_id

        return mask, scene_flow


    def check_occlusion(self):
        coords_world = []
        point_id_list = []
        cur_cnt = 0
        for obj_id in self.object_ids:
            position, orientation = p.getBasePositionAndOrientation(obj_id)
            coord = self._get_coord(obj_id, position, orientation)
            coords_world.append(coord)
            cur_cnt += 1
            point_id_list.append([cur_cnt for _ in range(coord.shape[0])])
        point_id = np.concatenate(point_id_list)
        coords_world = np.concatenate(coords_world)
        camera_view_matrix = np.array(self.sim.camera_params[1]['camera_view_matrix']).reshape(4, 4).T
        camera_intr = self.sim.camera_params[1]['camera_intr']
        image_size = self.sim.camera_params[1]['camera_image_size']
        coords_2d = project_pts_to_2d(coords_world.T, camera_view_matrix, camera_intr)

        y = np.round(coords_2d[0]).astype(np.int)
        x = np.round(coords_2d[1]).astype(np.int)
        depth = coords_2d[2]
        valid_idx = np.logical_and(
            np.logical_and(x >= 0, x < image_size[0]),
            np.logical_and(y >= 0, y < image_size[1])
        )
        x = x[valid_idx]
        y = y[valid_idx]
        depth = depth[valid_idx]
        point_id = point_id[valid_idx]

        sort_id = np.argsort(-depth)
        x = x[sort_id]
        y = y[sort_id]
        point_id = point_id[sort_id]

        mask = np.zeros([image_size[0], image_size[1]])
        mask[x, y] = point_id

        obj_num = len(self.object_ids)
        mask_sep = np.zeros([obj_num + 1, image_size[0], image_size[1]])
        mask_sep[point_id, x, y] = 1
        for i in range(obj_num):
            tot_pixel_num = np.sum(mask_sep[i + 1])
            vis_pixel_num = np.sum((mask == (i+1)).astype(np.float))
            if vis_pixel_num < 0.4 * tot_pixel_num:
                return False
        return True

    def _get_image_and_heightmap(self):
        color_image0, depth_image0 = self.sim.get_camera_data(self.sim.camera_params[0])
        color_image1, depth_image1 = self.sim.get_camera_data(self.sim.camera_params[1])
        color_image2, depth_image2 = self.sim.get_camera_data(self.sim.camera_params[2])

        color_heightmap, depth_heightmap = self.sim.get_heightmap(color_image2, depth_image2, self.sim.camera_params[2])

        self.current_depth_heightmap = depth_heightmap
        self.current_color_heightmap = color_heightmap
        self.current_depth_image0 = depth_image0
        self.current_color_image0 = color_image0
        self.current_depth_image1 = depth_image1
        self.current_color_image1 = color_image1

    def _random_drop(self, object_num, object_type):
        large_object_id = np.random.choice(object_num)
        if object_type == 'cube' or np.random.rand() < 0.1:
            large_object_id = -1
        self.large_object_id = large_object_id
        self.can_with_box = False

        while True:
            xy_pos = np.random.rand(object_num, 2) * 0.26 + np.asarray([0.5-0.13, -0.13])
            flag = True
            for i in range(object_num - 1):
                for j in range(i + 1, object_num):
                    d = np.sqrt(np.sum((xy_pos[i] - xy_pos[j])**2))
                    if i == large_object_id or j == large_object_id:
                        if d < 0.13:
                            flag = False
                    else:
                        if d < 0.07:
                            flag = False
            if large_object_id != -1 and xy_pos[large_object_id][1] > -0.05 and np.random.rand() < 0.15:
                flag = False
            if flag:
                break

        xy_pos -= np.mean(xy_pos, 0)
        xy_pos += np.array([0.5, 0])

        for i in range(object_num):
            if object_type == 'cube':
                md = np.ones([60, 60, 70])
                coord = (np.asarray(np.nonzero(md)).T + 0.5 - np.array([30, 30, 35]))
                size_cube = np.random.choice([700, 750, 800, 850, 900, 1000, 1100, 1200, 1400])
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array([30, 30, 35]) / size_cube)
                body_id = p.createMultiBody(
                    0.05, collision_id, -1,
                    [xy_pos[i, 0], xy_pos[i, 1], 0.2],
                    p.getQuaternionFromEuler(np.random.rand(3) * np.pi)
                )
                p.changeDynamics(body_id, -1, spinningFriction=0.003, lateralFriction=0.25, mass=0.05)
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))
                self.object_ids.append(body_id)
                self.voxel_coord[body_id] = coord / size_cube
                time.sleep(0.2)
            elif object_type == 'ycb':
                # get object
                if i == large_object_id:
                    obj_name = np.random.choice(self.ycb_info['large_list'])
                else:
                    obj_name = np.random.choice(self.ycb_info['normal_list'])

                with open(osp.join(self.ycb_path, obj_name, 'model_com.binvox'), 'rb') as f:
                    md = read_as_coord_array(f)
                coord = (md.data.T + 0.5) / md.dims * md.scale + md.translate

                # position & quat
                random_euler = [0, 0, np.random.rand() * 2 * np.pi]
                quat = p.getQuaternionFromEuler(random_euler)
                obj_position = [xy_pos[i, 0], xy_pos[i, 1], np.max(-coord[:, 2]) + 0.01]

                urdf_path = osp.join(self.ycb_path, obj_name, 'obj.urdf')
                body_id = p.loadURDF(
                    fileName=urdf_path,
                    basePosition=obj_position,
                    baseOrientation=quat,
                    globalScaling=1
                )
                p.changeDynamics(body_id, -1, spinningFriction=0.003, lateralFriction=0.25, mass=0.05)

                self.object_ids.append(body_id)
                self.voxel_coord[body_id] = coord
                time.sleep(2)
            elif (object_type=='shapenet' and i != large_object_id and np.random.rand() < 0.3) or \
                    (object_type=='shapenet' and i == large_object_id and np.random.rand() < 0.12):
                box_size = 'small'
                if i == large_object_id:
                    box_size='large'
                elif self.can_with_box and np.random.rand() < 0.1:
                    self.can_with_box=False
                    box_size='large'

                if box_size == 'large':
                    dim_x = np.random.choice(list(range(35, 55)))
                    dim_y = np.random.choice(list(range(70, 85)))
                    dim_z = np.random.choice(list(range(15, 30)))
                else:
                    dim_x = np.random.choice(list(range(25, 40)))
                    dim_y = np.random.choice(list(range(30, 60)))
                    dim_z = np.random.choice(list(range(15, 25)) + [30, 32])
                md = np.ones([dim_x, dim_y, dim_z])
                coord = (np.asarray(np.nonzero(md)).T + 0.5 - np.array([dim_x / 2, dim_y / 2, dim_z / 2]))
                size_cube = 500
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array(
                    [dim_x / 2, dim_y / 2, dim_z / 2]) / size_cube)
                body_id = p.createMultiBody(
                    0.05, collision_id, -1,
                    [xy_pos[i, 0], xy_pos[i, 1], 0.1],
                    [xy_pos[i, 0], xy_pos[i, 1], 0.1],
                    p.getQuaternionFromEuler([0, 0, np.random.rand() * np.pi])
                )
                p.changeDynamics(body_id, -1, spinningFriction=0.003, lateralFriction=0.25, mass=0.05)
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))
                self.object_ids.append(body_id)
                self.voxel_coord[body_id] = coord / size_cube
                time.sleep(0.2)
            else:
                object_cat_cur = np.random.choice(list(self.shapenet_info.keys()))
                if np.random.rand() < 0.3:
                    object_cat_cur = 'can'
                if i == large_object_id and object_cat_cur == 'can':
                    self.can_with_box=True

                category_id = self.shapenet_info[object_cat_cur]['category_id']
                tmp = np.random.choice(len(self.shapenet_info[object_cat_cur]['object_id']))
                object_id = self.shapenet_info[object_cat_cur]['object_id'][tmp]
                urdf_path = osp.join(self.shapenet_path, '%s/%s/obj.urdf' % (category_id, object_id))

                # load object
                if i == large_object_id:
                    scaling_range = self.shapenet_info[object_cat_cur]['large_scaling']
                else:
                    scaling_range = self.shapenet_info[object_cat_cur]['global_scaling']

                globalScaling = np.random.rand() * (scaling_range[1] - scaling_range[0]) + scaling_range[0]

                # save nonzero voxel coord
                with open(osp.join(self.shapenet_path, '%s/%s/model_com.binvox' % (category_id, object_id)),
                          'rb') as f:
                    md = read_as_coord_array(f)
                coord = (md.data.T + 0.5) / md.dims * md.scale + md.translate
                coord = coord * 0.15 * globalScaling  # 0.15 is the rescale value in .urdf

                # position & quat
                random_euler = [0, 0, np.random.rand() * 2 * np.pi]
                quat = p.getQuaternionFromEuler(random_euler)
                obj_position = [xy_pos[i, 0], xy_pos[i, 1], np.max(-coord[:, 2]) + 0.01]

                body_id = p.loadURDF(
                    fileName=urdf_path,
                    basePosition=obj_position,
                    baseOrientation=quat,
                    globalScaling=globalScaling
                )

                p.changeDynamics(body_id, -1, spinningFriction=0.003, lateralFriction=0.25, mass=0.05)
                p.changeVisualShape(body_id, -1, rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))
                self.object_ids.append(body_id)
                self.voxel_coord[body_id] = coord
                time.sleep(0.2)
        for obj_id in self.object_ids:
            self.cnt_dict[obj_id] = 0
            init_p = p.getBasePositionAndOrientation(obj_id)[0]
            self.init_position[obj_id] = np.asarray(init_p[:2])
            self.last_direction[obj_id] = None

    # for heightmap
    def coord2pixel(self, x_coord, y_coord):
        x_pixel = int((x_coord - self.view_bounds[0, 0]) / self.heightmap_pixel_size)
        y_pixel = int((y_coord - self.view_bounds[1, 0]) / self.heightmap_pixel_size)
        return x_pixel, y_pixel

    def pixel2coord(self, x_pixel, y_pixel):
        x_coord = x_pixel * self.heightmap_pixel_size + self.view_bounds[0, 0]
        y_coord = y_pixel * self.heightmap_pixel_size + self.view_bounds[1, 0]
        return x_coord, y_coord

    def policy_generation(self):
        def softmax(input):
            value = np.exp(input)
            output = value / np.sum(value)
            return output

        # choose object
        value = []
        for x in self.object_ids:
            t = self.cnt_dict[x]
            if t > 2:
                t = -2
            elif t > 1 and np.random.rand() < 0.5:
                t = -2
            self.cnt_dict[x] = t
            value.append(t)
        if self.large_object_id != -1:
            value[self.large_object_id] += 0.5
        obj_id = np.random.choice(self.object_ids, p=softmax(np.array(value)))

        # get position
        position = p.getBasePositionAndOrientation(obj_id)[0]
        position = np.asarray([position[0], position[1]])

        # choose direction
        direction_value = [0 for i in range(self.direction_num)]
        for d in range(self.direction_num):
            ang = 2 * np.pi * d / self.direction_num
            unit_vec = np.asarray([np.cos(ang), np.sin(ang)])

            off_direction = np.asarray([0.5, 0]) - position
            off_direction_unit = off_direction / np.sqrt(np.sum(off_direction ** 2))
            weight = 5 if self.last_direction[obj_id] is None else 1
            direction_value[d] += weight * np.sum(off_direction_unit * unit_vec) * np.exp(np.sum(np.abs(off_direction)))
            if np.sqrt(np.sum(off_direction ** 2)) > 0.2 and np.sum(off_direction_unit * unit_vec) < 0:
                direction_value[d] -= 10

            for obj_id_enm in self.object_ids:
                if obj_id_enm != obj_id:
                    obj_position_enm = p.getBasePositionAndOrientation(obj_id_enm)[0]
                    off_direction = np.asarray(obj_position_enm[:2]) - position
                    off_direction_unit = off_direction / np.sqrt(np.sum(off_direction ** 2))
                    if np.sqrt(np.sum(off_direction ** 2)) < 0.15 and np.sum(off_direction_unit * unit_vec) > 0.4:
                        direction_value[d] -= 3 * np.sum(off_direction_unit * unit_vec)
                    if np.sqrt(np.sum(off_direction ** 2)) < 0.15 and np.abs(
                            np.sum(off_direction_unit * unit_vec)) < 0.3:
                        direction_value[d] += 1.5

            off_direction = position - self.init_position[obj_id]
            if np.sum(np.abs(off_direction)) > 0.001:
                off_direction_unit = off_direction / np.sqrt(np.sum(off_direction ** 2))
                if np.sqrt(np.sum(off_direction ** 2)) < 0.25 and np.sum(off_direction_unit * unit_vec) < 0:
                    direction_value[d] += 1.5 * np.sum(off_direction_unit * unit_vec)

            if self.last_direction[obj_id] is not None:
                ang_last = 2 * np.pi * self.last_direction[obj_id] / self.direction_num
                unit_vec_last = np.asarray([np.cos(ang_last), np.sin(ang_last)])
                direction_value[d] += 2 * np.sum(unit_vec_last * unit_vec)

        direction = np.random.choice(self.direction_num, p=softmax(np.array(direction_value) / 2))

        direction_angle = direction / 4.0 * np.pi

        pos = position
        pos -= np.asarray([np.cos(direction_angle), np.sin(direction_angle)]) * 0.04

        for _ in range(5):
            x_coord, y_coord = pos[0], pos[1]
            x_pixel, y_pixel = self.coord2pixel(x_coord, y_coord)
            pos -= np.asarray([np.cos(direction_angle), np.sin(direction_angle)]) * 0.01
            if min(x_pixel, y_pixel) < 0 or max(x_pixel, y_pixel) >= 128:
                continue

            detection_mask = cv2.circle(np.zeros(self.heightmap_size), (x_pixel, y_pixel), 5, 1, thickness=-1)
            if np.max(self.current_depth_heightmap * detection_mask) > 0.005:
                continue

            d = 0.04
            new_pixel = self.coord2pixel(x_coord + np.cos(direction_angle) * d, y_coord + np.sin(direction_angle) * d)
            detection_mask = cv2.circle(np.zeros(self.heightmap_size), new_pixel, 3, 1, thickness=-1)
            if np.max(detection_mask * self.current_depth_heightmap) > 0.005:
                self.last_direction[obj_id] = direction
                self.cnt_dict[obj_id] += 1
                return x_pixel, y_pixel, x_coord, y_coord, 0.005, direction

        return None


    def poke(self):
        # log the current position & quat
        old_po_ors = [p.getBasePositionAndOrientation(object_id) for object_id in self.object_ids]
        output = self.get_scene_info()

        # generate action
        policy = None
        while policy is None:
            policy = self.policy_generation()

        x_pixel, y_pixel, x_coord, y_coord, z_coord, direction = policy

        # take action
        self.sim.primitive_push(
            position=[x_coord, y_coord, z_coord],
            rotation_angle=direction / 4.0 * np.pi,
            speed=0.005,
            distance=0.15
        )
        self.sim.robot_go_home()
        action = {'0': direction, '1': y_pixel, '2': x_pixel}

        mask_3d, scene_flow_3d = self._get_scene_flow_3d(old_po_ors)
        mask_2d, scene_flow_2d = self._get_scene_flow_2d(old_po_ors)

        output['action'] = action
        output['mask_3d'] = mask_3d
        output['scene_flow_3d'] = scene_flow_3d
        output['mask_2d'] = mask_2d
        output['scene_flow_2d'] = scene_flow_2d

        return output


    def get_scene_info(self, mask_info=False):
        self._get_image_and_heightmap()

        positions, orientations = [], []
        for i, obj_id in enumerate(self.object_ids):
            info = p.getBasePositionAndOrientation(obj_id)
            positions.append(info[0])
            orientations.append(info[1])

        scene_info = {
            'color_heightmap': self.current_color_heightmap,
            'depth_heightmap': self.current_depth_heightmap,
            'color_image': self.current_color_image0,
            'depth_image': self.current_depth_image0,
            'color_image_small': self.current_color_image1,
            'depth_image_small': self.current_depth_image1,
            'positions': np.array(positions),
            'orientations': np.array(orientations)
        }
        if mask_info:
            old_po_ors = [p.getBasePositionAndOrientation(object_id) for object_id in self.object_ids]
            mask_3d, scene_flow_3d = self._get_scene_flow_3d(old_po_ors)
            mask_2d, scene_flow_2d = self._get_scene_flow_2d(old_po_ors)
            scene_info['mask_3d'] = mask_3d
            scene_info['mask_2d'] = mask_2d

        return scene_info


    def reset(self, object_num=4, object_type=None):
        if object_type is None:
            object_type = self.object_type

        while True:
            # remove objects
            for obj_id in self.object_ids:
                p.removeBody(obj_id)
            self.object_ids = []
            self.cnt_dict = {}

            # load fences
            self.fence_id = p.loadURDF(
                fileName='assets/fence/tinker.urdf',
                basePosition=[0.5, 0, 0.001],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )

            # load objects
            self._random_drop(object_num, object_type)
            time.sleep(1)
            p.removeBody(self.fence_id)
            old_ps = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
            for _ in range(10):
                time.sleep(1)
                new_ps = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
                if np.sum((new_ps - old_ps) ** 2) < 1e-6:
                    break
                old_ps = new_ps
            self._get_image_and_heightmap()

            # check occlusion
            if self.check_occlusion():
                return


if __name__ == '__main__':
    env = SimulationEnv(gui_enabled=False)
    env.reset(4, 'ycb')

    # if you just want to get the information of the scene, use env.get_scene_info
    output = env.get_scene_info(mask_info=True)
    print(output.keys())
    
    # if use the pushing. env.poke() will also give you everything, together with scene flow
    output = env.poke()
    print(output.keys())