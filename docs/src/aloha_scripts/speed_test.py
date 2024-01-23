import pyagxrobots
import time
from matplotlib import pyplot as plt
import pyrealsense2 as rs

tracer = pyagxrobots.pysdkugv.TracerBase()
tracer.EnableCAN()

pipeline = rs.pipeline()
cfg = rs.config()
# if only pose stream is enabled, fps is higher (202 vs 30)
cfg.enable_stream(rs.stream.pose)
pipeline.start(cfg)

rs_vels = []
wheel_vels = []
target_vels = []

print('Start!')
for i in range(40):
    target_vel = 0.2
    target_vels.append(target_vel)
    # tracer.SetMotionCommand(linear_vel=target_vel)
    tracer.SetMotionCommand(angular_vel=target_vel)
    # wheel_vel = tracer.GetLinearVelocity()
    wheel_vel = tracer.GetAngularVelocity()
    wheel_vels.append(wheel_vel)

    frames = pipeline.wait_for_frames()
    pose_frame = frames.get_pose_frame()
    pose = pose_frame.get_pose_data()
    # rs_vel = pose.velocity.z
    rs_vel = pose.angular_velocity.y

    rs_vels.append(rs_vel)
    time.sleep(0.05)

print('End!')
pipeline.stop()
plt.plot(rs_vels, label='rs')
plt.plot(wheel_vels, label='wheel')
plt.plot(target_vels, label='target_vel')
plt.legend()
plt.savefig('vel.png')

