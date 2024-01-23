### Task parameters
import pathlib
import os

DATA_DIR = os.path.expanduser('~/data')
TASK_CONFIGS = {
    'aloha_mobile_dummy':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_dummy',
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # wash_pan
    'aloha_mobile_wash_pan':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_wash_pan',
        'episode_len': 1100,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wash_pan_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_wash_pan',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_wash_pan',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'episode_len': 1100,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # wipe_wine
    'aloha_mobile_wipe_wine':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_wipe_wine',
        'episode_len': 1300,
        'train_ratio': 0.9,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wipe_wine_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'episode_len': 1300,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wipe_wine_2':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_wipe_wine_2',
        'episode_len': 1300,
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_wipe_wine_2_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine_2',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_wipe_wine_2',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'episode_len': 1300,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # cabinet
    'aloha_mobile_cabinet':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
            DATA_DIR + '/aloha_mobile_cabinet_handles', # 200
            DATA_DIR + '/aloha_mobile_cabinet_grasp_pots', # 200
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
        ],
        'sample_weights': [6, 1, 1],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 1500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_cabinet_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
            DATA_DIR + '/aloha_mobile_cabinet_handles',
            DATA_DIR + '/aloha_mobile_cabinet_grasp_pots',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_cabinet',
        ],
        'sample_weights': [6, 1, 1, 2],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 1500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # elevator
    'aloha_mobile_elevator':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_elevator',
        'train_ratio': 0.99,
        'episode_len': 8500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_elevator_truncated':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2', # 1200
            DATA_DIR + '/aloha_mobile_elevator_button', # 800
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
        ],
        'sample_weights': [3, 3, 2],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 2250,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_elevator_truncated_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
            DATA_DIR + '/aloha_mobile_elevator_button',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
        ],
        'sample_weights': [3, 3, 2, 1],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 2250,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # high_five
    'aloha_mobile_high_five':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_high_five',
        'train_ratio': 0.9,
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_high_five_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_high_five',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_high_five',
        ],
        'sample_weights': [7.5, 2.5],
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # chair
    'aloha_mobile_chair':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_chair',
        'train_ratio': 0.95,
        'episode_len': 2400,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_chair_truncated':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_chair_truncated',
        'train_ratio': 0.95,
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_chair_truncated_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_chair_truncated',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_chair_truncated',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.95, # ratio of train data from the first dataset_dir
        'episode_len': 2000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    # shrimp
    'aloha_mobile_shrimp':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_shrimp',
        'train_ratio': 0.99,
        'episode_len': 4500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_shrimp_truncated':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_shrimp_truncated',
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 3750,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_shrimp_truncated_cotrain':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_shrimp_truncated',
            DATA_DIR + '/aloha_compressed_dataset',
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_shrimp_truncated',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 3750,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # 'aloha_mobile_shrimp_2_cotrain':{
    #     'dataset_dir': [
    #         DATA_DIR + '/aloha_mobile_shrimp_2',
    #         DATA_DIR + '/aloha_mobile_shrimp_before_spatula_down', # 2200
    #         DATA_DIR + '/aloha_compressed_dataset',
    #     ], # only the first dataset_dir is used for val
    #     'stats_dir': [
    #         DATA_DIR + '/aloha_mobile_shrimp_2',
    #     ],
    #     'sample_weights': [5, 3, 2],
    #     'train_ratio': 0.99, # ratio of train data from the first dataset_dir
    #     'episode_len': 4500,
    #     'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    # },
}

### ALOHA fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
