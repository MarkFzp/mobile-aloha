from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--all', action='store_true', default=False)
    args = argparser.parse_args()

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_right', init_node=False)

    all_bots = [puppet_bot_left, puppet_bot_right, master_bot_left, master_bot_right] if args.all else [puppet_bot_left, puppet_bot_right]
    master_bots = [master_bot_left, master_bot_right]
    for bot in all_bots:
        torque_on(bot)
    
    puppet_sleep_position = (0, -1.7, 1.55, 0, 0.65, 0)
    master_sleep_left_position = (-0.61, 0., 0.43, 0., 1.04, -0.65)
    master_sleep_right_position = (0.61, 0., 0.43, 0., 1.04, 0.65)
    all_positions = [puppet_sleep_position] * 2 + [master_sleep_left_position, master_sleep_right_position] if args.all else [puppet_sleep_position] * 2
    move_arms(all_bots, all_positions, move_time=2)

    if args.all:
        master_sleep_left_position_2 = (0., 0.66, -0.27, -0.0, 1.1, 0)
        master_sleep_right_position_2 = (0., 0.66, -0.27, -0.0, 1.1, 0)
        move_arms(master_bots, [master_sleep_left_position_2, master_sleep_right_position_2], move_time=1)



if __name__ == '__main__':
    main()
