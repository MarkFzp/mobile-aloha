import os
import time
import h5py
import argparse
import h5py_cache
import numpy as np
from tqdm import tqdm
import cv2

from constants import DT, START_ARM_POSE, TASK_CONFIGS, FPS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS

import IPython
e = IPython.embed


def opening_ceremony(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    torque_on(puppet_bot_left)
    torque_on(master_bot_left)
    torque_on(puppet_bot_right)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot_left, puppet_bot_left, master_bot_right, puppet_bot_right], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -1.4
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print(f'Dataset name: {dataset_name}')

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    time0 = time.time()
    DT = 1 / FPS
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        action = get_action(master_bot_left, master_bot_right)
        t1 = time.time() #
        ts = env.step(action)
        t2 = time.time() #
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))
    print(f'Avg fps: {max_timesteps / (time.time() - time0)}')

    # Torque on both master bots
    torque_on(master_bot_left)
    torque_on(master_bot_right)
    # Open puppet grippers
    env.puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    env.puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f'\n\nfreq_mean is {freq_mean}, lower than 30, re-collecting... \n\n\n\n')
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    
    action                  (14,)         'float64'
    base_action             (2,)          'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
        # '/base_action_t265': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
    
    # plot /base_action vs /base_action_t265
    # import matplotlib.pyplot as plt
    # plt.plot(np.array(data_dict['/base_action'])[:, 0], label='base_action_linear')
    # plt.plot(np.array(data_dict['/base_action'])[:, 1], label='base_action_angular')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 0], '--', label='base_action_t265_linear')
    # plt.plot(np.array(data_dict['/base_action_t265'])[:, 1], '--', label='base_action_t265_angular')
    # plt.legend()
    # plt.savefig('record_episodes_vel_debug.png', dpi=300)

    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
            else:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))
        _ = root.create_dataset('base_action', (max_timesteps, 2))
        # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len

    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('right', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args())) # TODO
    # debug()


