import pyagxrobots
import time
import IPython
e = IPython.embed

import pyrealsense2 as rs
from pyquaternion import Quaternion
import numpy as np
from simple_pid import PID
import argparse
np.set_printoptions(precision=3, suppress=True)

argparser = argparse.ArgumentParser()
argparser.add_argument('--debug', action='store_true')
args = argparser.parse_args()

# setup base
tracer = pyagxrobots.pysdkugv.TracerBase()
tracer.EnableCAN()

# setup realsense
pipeline = rs.pipeline()
cfg = rs.config()
# if only pose stream is enabled, fps is higher (202 vs 30)
cfg.enable_stream(rs.stream.pose)
pipeline.start(cfg)

def yaw_to_vector(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)])

def get_pose():
    frames = pipeline.wait_for_frames()
    pose_frame = frames.get_pose_frame()
    pose = pose_frame.get_pose_data()
    yaw = -1 * Quaternion(pose.rotation.w, pose.rotation.x, pose.rotation.y, pose.rotation.z).yaw_pitch_roll[0]
    pose_np = np.array([pose.translation.z, pose.translation.x, yaw])

    return pose_np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
    def compute(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

# init global coords
r = 0.2
start_pose = get_pose()
delta_target_pose_l = [
    np.array([
        r * np.sin(theta), 
        -(r - r * np.cos(theta)), 
        -theta]) for theta in np.linspace(0, np.pi / 2, int((np.pi / 2) // (0.01 / r)))
]
target_pose_l = []
for delta_target_pose in delta_target_pose_l:
    transformed_delta_target_pose_x = delta_target_pose[0] * np.cos(start_pose[2]) - delta_target_pose[1] * np.sin(start_pose[2])
    transformed_delta_target_pose_y = delta_target_pose[0] * np.sin(start_pose[2]) + delta_target_pose[1] * np.cos(start_pose[2])
    transformed_target_pose = np.array([
        start_pose[0] + transformed_delta_target_pose_x,
        start_pose[1] + transformed_delta_target_pose_y,
        normalize_angle(start_pose[2] + delta_target_pose[2])
    ])
    target_pose_l.append(transformed_target_pose)


MAX_LINEAR_VEL = 0.1
MIN_LINEAR_VEL = -0.1
MAX_ANGULAR_VEL = 0.3
MIN_ANGULAR_VEL = -0.3
POS_THRESHOLD = 0.05
ORN_THRESHOLD = 0.1
DT = 0.1

print(target_pose_l)
for target_pose in target_pose_l:
    target_pos = target_pose[:2]
    target_orn = target_pose[2]

    while True:
        curr_pose = get_pose()
        curr_pos, curr_orn = curr_pose[:2], curr_pose[2]

        distance_to_target = np.linalg.norm(target_pos - curr_pos) 
        error_orn = normalize_angle(target_orn - curr_orn)
        if distance_to_target < POS_THRESHOLD and abs(error_orn) < ORN_THRESHOLD:
            break
        
        target_heading = np.arctan2(target_pos[1] - curr_pos[1], target_pos[0] - curr_pos[0])
        error_heading = normalize_angle(target_heading - curr_orn)

        if distance_to_target < POS_THRESHOLD:
            is_forward = 1
            v = 0
            w = error_orn / DT
        else:
            is_forward = np.sign(np.dot(
                yaw_to_vector(target_heading), yaw_to_vector(curr_orn)))
            v = distance_to_target * is_forward / DT
            w = error_heading * is_forward / DT

        v = np.clip(v, MIN_LINEAR_VEL, MAX_LINEAR_VEL)
        w = np.clip(w, MIN_ANGULAR_VEL, MAX_ANGULAR_VEL)

        print(f'''
            curr_pose: {curr_pose},
            target_pose: {target_pose},
            is_forward: {is_forward},
            action: {np.array([v, w])},
            error_heading: {error_heading:0.3f},
            error_orn: {error_orn:0.3f},
            distance: {distance_to_target:0.3f}
            --------------------------
        ''')

        if args.debug:
            tracer.SetMotionCommand(
                linear_vel=0,
                angular_vel=0.1
            )
            time.sleep(1)
        else:
            # set motion command
            tracer.SetMotionCommand(
                linear_vel=v,
                angular_vel=w
            )
            time.sleep(0.05)



print('End!')
