import os
import h5py
from robot_utils import move_grippers
import argparse
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]

    env = make_real_env(init_node=True)
    env.reset()
    for action in actions:
        env.step(action)

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))


