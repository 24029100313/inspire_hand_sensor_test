# Install script for directory: /home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/install/urdf_left_with_force_sensor")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/config" TYPE DIRECTORY FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/config/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/launch" TYPE DIRECTORY FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/launch/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/meshes" TYPE DIRECTORY FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/meshes/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/urdf" TYPE DIRECTORY FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/urdf/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/rviz" TYPE DIRECTORY FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/rviz/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/urdf_left_with_force_sensor" TYPE PROGRAM FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/script/glove_hand_control.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/urdf_left_with_force_sensor")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/urdf_left_with_force_sensor")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/environment" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/environment" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_index/share/ament_index/resource_index/packages/urdf_left_with_force_sensor")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor/cmake" TYPE FILE FILES
    "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_core/urdf_left_with_force_sensorConfig.cmake"
    "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/ament_cmake_core/urdf_left_with_force_sensorConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/urdf_left_with_force_sensor" TYPE FILE FILES "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/lr-2002/project/utpt/inspire_hand/ros2/urdf_left_with_force_sensor/build/urdf_left_with_force_sensor/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
