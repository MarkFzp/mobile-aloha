import pyrealsense2 as rs
import time
from pprint import pprint
from collections import namedtuple
from functools import partial
from matplotlib import pyplot as plt

attrs = [
    'acceleration',
    'angular_acceleration',
    'angular_velocity',
    'mapper_confidence',
    'rotation',
    'tracker_confidence',
    'translation',
    'velocity',
    ]
Pose = namedtuple('Pose', attrs)

def main():
    pipeline = rs.pipeline()
    cfg = rs.config()
    # if only pose stream is enabled, fps is higher (202 vs 30)
    cfg.enable_stream(rs.stream.pose)
    pipeline.start(cfg)
    poses = []
    z_vels = []
    x_vels = []
    y_vels = []

    try:
        print('Start!')
        while True:
            frames = pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()
            if pose_frame:
                pose = pose_frame.get_pose_data()
                n = pose_frame.get_frame_number()
                timestamp = pose_frame.get_timestamp()
                p = Pose(*map(partial(getattr, pose), attrs))
                z_vel = pose.velocity.z
                y_vel = pose.velocity.y
                x_vel = pose.velocity.x
                z_vels.append(z_vel)
                x_vels.append(x_vel)
                y_vels.append(y_vel)

                poses.append((n, timestamp, p))
                if len(poses) == 1000:
                    return
            time.sleep(0.02)
    finally:
        print('End!')
        pipeline.stop()
        duration = (poses[-1][1]-poses[0][1])/1000
        print(f'start: {poses[0][1]}')
        print(f'end:   {poses[-1][1]}')
        print(f'duration: {duration}s')
        print(f'fps: {len(poses)/duration}')
        plt.plot(z_vels, label='z_vel')
        plt.plot(x_vels, label='x_vel')
        plt.plot(y_vels, label='y_vel')
        plt.legend()
        plt.savefig('rs_vel.png')
        plt.show()


if __name__ == "__main__":
    main()