cmake_minimum_required(VERSION 2.8.3)
project(aloha)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  interbotix_xs_msgs
  interbotix_xsarm_control
  rosbag
  roscpp
  tf2_ros
  rviz
  sensor_msgs
  std_msgs
  std_srvs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
catkin_package(
 CATKIN_DEPENDS interbotix_xs_msgs interbotix_xsarm_control rosbag roscpp tf2_ros rviz sensor_msgs std_msgs std_srvs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
)


