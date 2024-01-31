import os
import h5py
import numpy as np
from robot_utils import move_grippers, calibrate_linear_vel, smooth_base_action, postprocess_base_action
import argparse
import matplotlib.pyplot as plt
from real_env import make_real_env
from constants import JOINT_NAMES, PUPPET_GRIPPER_JOINT_OPEN, FPS
import time

import IPython
e = IPython.embed

STATE_NAMES = JOINT_NAMES + ["gripper", 'left_finger', 'right_finger']

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'
    actuator_network_dir = args['actuator_network_dir']
    history_len = args['history_len']
    future_len = args['future_len']
    prediction_len = args['prediction_len']
    use_actuator_net = actuator_network_dir is not None

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]
        base_actions = root['/base_action'][()]

    if use_actuator_net:
        from train_actuator_network import ActuatorNetwork
        import torch
        import pickle
        actuator_network = ActuatorNetwork(prediction_len)
        actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
        loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
        actuator_network.eval()
        actuator_network.cuda()
        print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

        actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
        with open(actuator_stats_path, 'rb') as f:
            actuator_stats = pickle.load(f)

        norm_observed_speed = (base_actions - actuator_stats["observed_speed_mean"]) / actuator_stats["observed_speed_std"]
        out_unnorm_fn = lambda x: (x * actuator_stats["commanded_speed_std"]) + actuator_stats["commanded_speed_mean"]

        history_pad = np.zeros((history_len, 2))
        future_pad = np.zeros((future_len, 2))
        norm_observed_speed = np.concatenate([history_pad, norm_observed_speed, future_pad], axis=0)

        episode_len = base_actions.shape[0]
        assert(episode_len % prediction_len == 0)

        processed_base_actions = []
        for t in range(0, episode_len, prediction_len):
            offset_start_ts = t + history_len
            actuator_net_in = norm_observed_speed[offset_start_ts-history_len: offset_start_ts+future_len]
            actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
            pred = actuator_network(actuator_net_in)
            pred = pred.detach().cpu().numpy()[0]
            processed_base_actions += out_unnorm_fn(pred).tolist()
        
        processed_base_actions = np.array(processed_base_actions)
        assert processed_base_actions.shape == base_actions.shape

        plt.plot(base_actions[:, 0], label='action_linear')
        plt.plot(processed_base_actions[:, 0], '--', label='processed_action_linear')
        plt.plot(base_actions[:, 1], label='action_angular')
        plt.plot(processed_base_actions[:, 1], '--', label='processed_action_angular')
        plt.plot()
        plt.legend()
        plt.show()
    else:
        # processed_base_actions = smooth_base_action(base_actions)
        processed_base_actions = base_actions

    env = make_real_env(init_node=True, setup_base=True)
    env.reset()
    obs_wheels = []
    obs_tracer = []

    time0 = time.time()
    DT = 1 / FPS
    for action, base_action in zip(actions, processed_base_actions):
        time1 = time.time()
        # base_action = calibrate_linear_vel(base_action, c=0.19)
        # base_action = postprocess_base_action(base_action)
        ts = env.step(action, base_action, get_tracer_vel=True)
        obs_wheels.append(ts.observation['base_vel'])
        obs_tracer.append(ts.observation['tracer_vel'])
        time.sleep(max(0, DT - (time.time() - time1)))
    print(f'Avg fps: {len(actions) / (time.time() - time0)}')
    obs_wheels = np.array(obs_wheels)
    obs_tracer = np.array(obs_tracer)

    plt.plot(base_actions[:, 0], label='action_linear')
    plt.plot(processed_base_actions[:, 0], '--', label='processed_action_linear')
    plt.plot(obs_wheels[:, 0], '--', label='obs_wheels_linear')
    plt.plot(obs_tracer[:, 0], '-.', label='obs_tracer_linear')
    plt.plot()
    plt.legend()
    plt.savefig('replay_episodes_linear_vel.png', dpi=300)
    
    plt.clf()
    plt.plot(base_actions[:, 1], label='action_angular')
    plt.plot(processed_base_actions[:, 1], '--', label='processed_action_angular')
    plt.plot(obs_wheels[:, 1], '--', label='obs_wheels_angular')
    plt.plot(obs_tracer[:, 1], '-.', label='obs_tracer_angular')
    plt.plot()
    plt.legend()
    plt.savefig('replay_episodes_angular_vel.png', dpi=300)

    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)
    main(vars(parser.parse_args()))


