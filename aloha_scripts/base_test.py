import pyagxrobots
import time
import IPython
e = IPython.embed

tracer = pyagxrobots.pysdkugv.TracerBase()
tracer.EnableCAN()


while True:
    curr_vel = tracer.GetLinearVelocity()

    if abs(curr_vel) >= 0.003 and abs(curr_vel) <= 0.009: # somewhat moving
        set_vel = curr_vel/abs(curr_vel) * 0.009
    else:
        set_vel = curr_vel
    
    if abs(set_vel) < 0.02:
        set_vel *= 1.3
    else:
        set_vel *= 0.8

    print("curr_vel: ", curr_vel, "set_vel: ", set_vel)
    # tracer.SetMotionCommand(linear_vel=set_vel)
    time.sleep(0.02)
