cmake_minimum_required(VERSION 3.0.2)
project(add_jeff)

add_definitions(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_ros
	roscpp
	std_msgs
)

find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")


add_library(model_plugin SHARED model_plugin.cc)
target_link_libraries(model_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES})


catkin_package(
  DEPENDS
		rospy
		gazebo_ros
		roscpp
)

catkin_install_python(PROGRAMS src/talker.py src/listener.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
