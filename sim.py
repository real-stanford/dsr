import math
import threading
import time

import numpy as np
import pybullet as p
import pybullet_data

import utils


class PybulletSim:
    def __init__(self, gui_enabled, heightmap_pixel_size=0.004, tool='stick'):

        self._workspace_bounds = np.array([[0.244, 0.756],
                                           [-0.256, 0.256],
                                           [0.0, 0.192]])

        self._view_bounds = self._workspace_bounds

        # Start PyBullet simulation
        if gui_enabled:
            self._physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        else:
            self._physics_client = p.connect(p.DIRECT)  # non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        step_sim_thread = threading.Thread(target=self.step_simulation)
        step_sim_thread.daemon = True
        step_sim_thread.start()

        # Add ground plane & table
        self._plane_id = p.loadURDF("plane.urdf")
        # self._table_id = p.loadURDF('assets/table/table.urdf', [0.5, 0, 0], useFixedBase=True)

        # Add UR5 robot
        self._robot_body_id = p.loadURDF("assets/ur5/ur5.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
        self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._joint_epsilon = 0.01  # joint position threshold in radians for blocking calls (i.e. move until joint difference < epsilon)

        # Move robot to home joint configuration
        self._robot_home_joint_config = [-3.186603833231106, -2.7046623323544323, 1.9797780717750348,
                                         -0.8458013020952369, -1.5941890970134802, -0.04501555880643846]
        self.move_joints(self._robot_home_joint_config, blocking=True, speed=1.0)

        self.tool=tool
        # Attach a sticker to UR5 robot
        self._gripper_body_id = p.loadURDF("assets/stick/stick.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [0.5, 0.1, 0.2],
                                        p.getQuaternionFromEuler([np.pi, 0, 0]))
        self._robot_tool_joint_idx = 9
        self._robot_tool_tip_joint_idx = 10
        self._robot_tool_offset = [0, 0, -0.0725]

        p.createConstraint(self._robot_body_id, self._robot_tool_joint_idx, self._gripper_body_id, 0,
                           jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=self._robot_tool_offset,
                           childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        self._tool_tip_to_ee_joint = [0, 0, 0.17]
        # Define Denavit-Hartenberg parameters for UR5
        self._ur5_kinematics_d = np.array([0.089159, 0., 0., 0.10915, 0.09465, 0.0823])
        self._ur5_kinematics_a = np.array([0., -0.42500, -0.39225, 0., 0., 0.])

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(
                self._gripper_body_id, i,
                lateralFriction=1.0,
                spinningFriction=1.0,
                rollingFriction=0.0001,
                frictionAnchor=True
            )

        # Add RGB-D camera (mimic RealSense D415)
        self.camera_params = {
            # large camera, image_size = (240 * 4, 320 * 4)
            0: self._get_camera_param(
                camera_position=[0.5, -0.7, 0.3],
                camera_image_size=[240 * 4, 320 * 4]
            ),
            # small camera, image_size = (240, 320)
            1: self._get_camera_param(
                camera_position=[0.5, -0.7, 0.3],
                camera_image_size=[240, 320]
            ),
            # top-down camera, image_size = (480, 480)
            2: self._get_camera_param(
                camera_position=[0.5, 0, 0.5],
                camera_image_size=[480, 480]
            ),
        }


        self._heightmap_pixel_size = heightmap_pixel_size
        self._heightmap_size = np.round(
            ((self._view_bounds[1][1] - self._view_bounds[1][0]) / self._heightmap_pixel_size,
             (self._view_bounds[0][1] - self._view_bounds[0][0]) / self._heightmap_pixel_size)).astype(int)


    def _get_camera_param(self, camera_position, camera_image_size):
        camera_lookat = [0.5, 0, 0]
        camera_up_direction = [0, camera_position[2], -camera_position[1]]
        camera_view_matrix = p.computeViewMatrix(camera_position, camera_lookat, camera_up_direction)
        camera_pose = np.linalg.inv(np.array(camera_view_matrix).reshape(4, 4).T)
        camera_pose[:, 1:3] = -camera_pose[:, 1:3]
        camera_z_near = 0.01
        camera_z_far = 10.0
        camera_fov_w = 69.40
        camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
        camera_fov_h = (math.atan((float(camera_image_size[0]) / 2) / camera_focal_length) * 2 / np.pi) * 180
        camera_projection_matrix = p.computeProjectionMatrixFOV(
            fov=camera_fov_h,
            aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
            nearVal=camera_z_near,
            farVal=camera_z_far
        )  # notes: 1) FOV is vertical FOV 2) aspect must be float
        camera_intrinsics = np.array(
            [[camera_focal_length, 0, float(camera_image_size[1]) / 2],
             [0, camera_focal_length, float(camera_image_size[0]) / 2],
             [0, 0, 1]])
        camera_param = {
            'camera_image_size': camera_image_size,
            'camera_intr': camera_intrinsics,
            'camera_pose': camera_pose,
            'camera_view_matrix': camera_view_matrix,
            'camera_projection_matrix': camera_projection_matrix,
            'camera_z_near': camera_z_near,
            'camera_z_far': camera_z_far
        }
        return camera_param

    # Step through simulation time
    def step_simulation(self):
        while True:
            p.stepSimulation()
            time.sleep(0.0001)

    # Get RGB-D heightmap from RGB-D image
    def get_heightmap(self, color_image, depth_image, cam_param):
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img=color_image,
            depth_img=depth_image,
            cam_intrinsics=cam_param['camera_intr'],
            cam_pose=cam_param['camera_pose'],
            workspace_limits=self._view_bounds,
            heightmap_resolution=self._heightmap_pixel_size
        )
        return color_heightmap, depth_heightmap

    # Get latest RGB-D image
    def get_camera_data(self, cam_param):
        camera_data = p.getCameraImage(cam_param['camera_image_size'][1], cam_param['camera_image_size'][0],
                                       cam_param['camera_view_matrix'], cam_param['camera_projection_matrix'],
                                       shadow=1, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)

        color_image = np.asarray(camera_data[2]).reshape(
            [cam_param['camera_image_size'][0], cam_param['camera_image_size'][1], 4])[:, :, :3]  # remove alpha channel
        z_buffer = np.asarray(camera_data[3]).reshape(cam_param['camera_image_size'])
        camera_z_near = cam_param['camera_z_near']
        camera_z_far = cam_param['camera_z_far']
        depth_image = (2.0 * camera_z_near * camera_z_far) / (
                camera_z_far + camera_z_near - (2.0 * z_buffer - 1.0) * (
                camera_z_far - camera_z_near))
        return color_image, depth_image

    # Move robot tool to specified pose
    def move_tool(self, position, orientation, blocking=False, speed=0.03):

        # Use IK to compute target joint configuration
        target_joint_state = np.array(
            p.calculateInverseKinematics(self._robot_body_id, self._robot_tool_tip_joint_idx, position, orientation,
                                         maxNumIterations=10000,
                                         residualThreshold=.0001))
        target_joint_state[5] = (
                (target_joint_state[5] + np.pi) % (2 * np.pi) - np.pi)  # keep EE joint angle between -180/+180

        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state,
                                    positionGains=speed * np.ones(len(self._robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):  # and (time.time()-timeout_t0) < timeout:
                if time.time() - timeout_t0 > 5:
                    p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                                self._robot_home_joint_config,
                                                positionGains=np.ones(len(self._robot_joint_indices)))
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
                time.sleep(0.001)

    # Move robot arm to specified joint configuration
    def move_joints(self, target_joint_state, blocking=False, speed=0.03):

        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices,
                                    p.POSITION_CONTROL, target_joint_state,
                                    positionGains=speed * np.ones(len(self._robot_joint_indices)))

        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):
                if time.time() - timeout_t0 > 5:
                    p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                                self._robot_home_joint_config,
                                                positionGains=np.ones(len(self._robot_joint_indices)))
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
                time.sleep(0.001)


    def robot_go_home(self, blocking=True, speed=0.1):
        self.move_joints(self._robot_home_joint_config, blocking, speed)


    def primitive_push(self, position, rotation_angle, speed=0.01, distance=0.1):
        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [push_orientation[0] * np.cos(rotation_angle) - push_orientation[1] * np.sin(rotation_angle),
             push_orientation[0] * np.sin(rotation_angle) + push_orientation[1] * np.cos(rotation_angle), 0.0])
        target_x = position[0] + push_direction[0] * distance
        target_y = position[1] + push_direction[1] * distance
        position_end = np.asarray([target_x, target_y, position[2]])
        self.move_tool([position[0], position[1], 0.15], orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.05)
        self.move_tool(position, orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.1)
        self.move_tool(position_end, orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=speed)

        position_end[2]=0.15
        self.move_tool(position_end, orientation=[-1.0, 1.0, 0.0, 0.0], blocking=True, speed=0.005)
