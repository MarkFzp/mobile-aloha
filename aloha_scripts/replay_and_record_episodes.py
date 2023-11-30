import os
import h5py
import numpy as np
from robot_utils import move_grippers, calibrate_linear_vel, smooth_base_action, postprocess_base_action
import argparse
import matplotlib.pyplot as plt
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN, fps
import time

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']

def store_new_dataset(input_dataset_path, output_dataset_path, obs_wheels, obs_tracer):
    # Check if output path exists
    if os.path.exists(output_dataset_path):
        print(f"The file {output_dataset_path} already exists. Exiting...")
        return

    # Load the uncompressed dataset
    with h5py.File(input_dataset_path, 'r') as infile:
        # Create the replayed dataset
        with h5py.File(output_dataset_path, 'w') as outfile:

            outfile.attrs['sim'] = infile.attrs['sim']
            outfile.attrs['compress'] = True

            # Copy non-image data directly
            for key in infile.keys():
                outfile.copy(infile[key], key)

            max_timesteps = infile['action'].shape[0]
            _ = outfile.create_dataset('obs_wheels', (max_timesteps, 2))
            _ = outfile.create_dataset('obs_tracer', (max_timesteps, 2))

            outfile['obs_wheels'][()] = obs_wheels
            outfile['obs_tracer'][()] = obs_tracer

    print(f"Replayed dataset saved to {output_dataset_path}")


def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    dataset_new_path = os.path.join(dataset_dir, dataset_name + '_replayed.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]
        base_actions = root['/base_action'][()]

    env = make_real_env(init_node=True, setup_base=True)
    env.reset()
    # base_actions = smooth_base_action(base_actions)
    obs_wheels = []
    obs_tracer = []

    offset = 0
    scale = 1
    apply_actions = actions
    apply_base_actions = base_actions[offset:] * scale

    DT = 1 / fps
    for action, base_action in zip(apply_actions, apply_base_actions):
        time1 = time.time()
        # base_action = calibrate_linear_vel(base_action, c=0.19)
        # base_action = postprocess_base_action(base_action)
        ts = env.step(action, base_action, get_tracer_vel=True)
        obs_wheels.append(ts.observation['base_vel'])
        obs_tracer.append(ts.observation['tracer_vel'])
        time.sleep(max(0, DT - (time.time() - time1)))
    obs_wheels = np.array(obs_wheels)
    obs_tracer = np.array(obs_tracer)

    store_new_dataset(dataset_path, dataset_new_path, obs_wheels, obs_tracer)

    plt.plot(base_actions[:, 0], label='action_linear')
    plt.plot(base_actions[:, 1], label='action_angular')
    plt.plot(obs_wheels[:, 0], '--', label='obs_wheels_linear')
    plt.plot(obs_wheels[:, 1], '--', label='obs_wheels_angular')
    plt.plot(obs_tracer[:, 0], '-.', label='obs_tracer_linear')
    plt.plot(obs_tracer[:, 1], '-.', label='obs_tracer_angular')
    plt.legend()
    plt.savefig('replay_and_record_episodes_vel_debug.png', dpi=300)
    

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))


