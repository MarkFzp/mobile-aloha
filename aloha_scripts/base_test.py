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
start_pose = get_pose()
delta_target_pose_l = [
    np.array([0.5, 0, 0]),
    np.array([0.5, 0, np.pi / 4]),
]
target_pose_l = []
for delta_target in delta_target_pose_l:
    target_pose_l.append(start_pose + delta_target)

orn_controller = PIDController(kp=1.0, ki=0, kd=0.05)

MAX_LINEAR_VEL = 0.1
MIN_LINEAR_VEL = -0.1
MAX_ANGULAR_VEL = 0.5
MIN_ANGULAR_VEL = -0.5
POS_THRESHOLD = 0.03
ORN_THRESHOLD = 0.1

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
        error_orn = normalize_angle(target_orn - curr_orn)

        # blend between heading and orientation error
        blend_factor = min(1, distance_to_target / (POS_THRESHOLD * 3))
        error = blend_factor * error_heading + (1 - blend_factor) * error_orn

        # compute angular velocity
        action_angular_vel = orn_controller.compute(error)
        action_angular_vel = np.clip(action_angular_vel, MIN_ANGULAR_VEL, MAX_ANGULAR_VEL)

        # compute linear velocity
        action_linear_vel = 0.5 * distance_to_target * np.cos(error_heading)
        action_linear_vel = np.clip(action_linear_vel, MIN_LINEAR_VEL, MAX_LINEAR_VEL)

        if args.debug:
            time.sleep(1)
            tracer.SetMotionCommand(
                linear_vel=0,
                angular_vel=0.1
            )
        else:
            # set motion command
            tracer.SetMotionCommand(
                linear_vel=action_linear_vel,
                angular_vel=action_angular_vel
            )

            time.sleep(0.05)

        print(f'''
            curr_pose: {curr_pose},
            target_pose: {target_pose},
            action: {np.array([action_linear_vel, action_angular_vel])}
            error: {np.array(error)},
            error_heading: {np.array(error_heading)},
            error_orn: {np.array(error_orn)},
            blend_factor: {np.array(blend_factor)}
            --------------------------
        ''')

print('End!')
