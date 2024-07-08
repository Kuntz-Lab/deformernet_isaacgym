cmake_minimum_required(VERSION 2.8.3)
project(point_cloud_segmentation)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  genmsg
  image_transport
  message_filters
  sensor_msgs
  geometry_msgs
  pcl_ros
  tf
  message_generation
)

## Generate services in the 'srv' folder
add_message_files(
  DIRECTORY msg
  FILES
  GraspObject.msg
  )


add_service_files(
  DIRECTORY srv
  FILES
  FilterPCL.srv
  SegmentGraspObject.srv
  GetVisualData.srv
  )


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  )


catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs roscpp rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(tabletop_object_segmenter src/table_object_segmenter.cpp)
add_dependencies(tabletop_object_segmenter point_cloud_segmentation_generate_messages_cpp)
target_link_libraries(tabletop_object_segmenter ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

