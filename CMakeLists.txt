cmake_minimum_required(VERSION 3.8)
project(vacancy_detector)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(table_drop_tf_node src/table_drop_tf_node.cpp)
add_executable(container_drop_tf_node src/container_drop_tf_node.cpp)
add_executable(shelf_drop_tf_node src/shelf_drop_tf_node.cpp)

install(TARGETS
  table_drop_tf_node
  container_drop_tf_node
  shelf_drop_tf_node
  DESTINATION lib/${PROJECT_NAME})

ament_target_dependencies(table_drop_tf_node
  rclcpp
  sensor_msgs
  std_srvs
  pcl_conversions
  visualization_msgs
  geometry_msgs
  tf2_ros
  Eigen3
)

ament_target_dependencies(container_drop_tf_node
  rclcpp
  sensor_msgs
  std_srvs
  pcl_conversions
  visualization_msgs
  geometry_msgs
  tf2_ros
  Eigen3
)

ament_target_dependencies(shelf_drop_tf_node
  rclcpp
  sensor_msgs
  std_srvs
  pcl_conversions
  visualization_msgs
  geometry_msgs
  tf2_ros
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
