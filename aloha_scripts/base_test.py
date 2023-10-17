import pyagxrobots
import time
import IPython
e = IPython.embed

import pyrealsense2 as rs
from pyquaternion import Quaternion
import numpy as np
from simple_pid import PID

tracer = pyagxrobots.pysdkugv.TracerBase()
tracer.EnableCAN()

pipeline = rs.pipeline()
cfg = rs.config()
# if only pose stream is enabled, fps is higher (202 vs 30)
cfg.enable_stream(rs.stream.pose)
pipeline.start(cfg)

def get_pos():
    frames = pipeline.wait_for_frames()
    pose_frame = frames.get_pose_frame()
    pose = pose_frame.get_pose_data()
    np_pos = np.array([pose.translation.x, pose.translation.y, pose.translation.z])

    return np_pos


delta_target_pos = (0, 0, 0.5)
start_pos = get_pos()
target_pos = start_pos + delta_target_pos

pid = PID(1, 0, 0.05, setpoint=target_pos[2])

MAX_LINEAR_SPEED = 0.1
MIN_LINEAR_SPEED = -0.1

while True:
    curr_pos = get_pos()
    action = pid(curr_pos[2])
    action = np.clip(action, MIN_LINEAR_SPEED, MAX_LINEAR_SPEED)
    
    tracer.SetMotionCommand(linear_vel=action)

    time.sleep(0.05)

    print(f'curr_pos: {curr_pos[2]}, target_pos: {target_pos[2]}, action: {action}')

