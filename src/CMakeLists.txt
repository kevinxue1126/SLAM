#cmake Version restrictions
cmake_minimum_required(VERSION 2.8)
#Construction name
project(ORB_SLAM2_pc)

SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
# With this option, cmake generates a JSON file with include paths to generate a JSON compilation database. 
# The compiled database JSON file is generated when cmake is executed, not when make is compiled.

#Compile mode Use IF(NOT ) ENDIF to place duplicate settings
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# Display compilation mode information
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# add-on module cmakeList.txt
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# opencv 
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

 #  Eigen3
find_package(Eigen3 3.1.0 REQUIRED)
# gui Pangolin
find_package(Pangolin REQUIRED)

# adding for point cloud viewer and mapper
find_package( PCL 1.7 REQUIRED )


# include library files
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}# gui
${PCL_INCLUDE_DIRS}#point cloud library
${PROJECT_SOURCE_DIR}/Examples/RGB-D/ty/common/include # Graph header file
${PROJECT_SOURCE_DIR}/Thirdparty/ncnn/include          # ncnn 
)

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

# Self-built library generation path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ncnn 
#set(NCNN_SOURCES
#${PROJECT_SOURCE_DIR}/Thirdparty/ncnn/ncnn_dect.cpp
#)
#add_library(ncnn_common STATIC ${NCNN_SOURCES}) 

set(NCNN_TARGET_LIB ncnn) # Camera Gallery

# Create a shared library SHARED dynamic link library
add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/pointcloudmapping.cc
${PROJECT_SOURCE_DIR}/Thirdparty/ncnn/ncnn_dect.cpp
)

# link library file
target_link_libraries(
${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PCL_LIBRARIES}
${NCNN_TARGET_LIB}
-fopenmp
)

#Executables are placed in the /bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


set(TY_TARGET_LIB tycam) # camera library
# ========================================
# === common, pcl is too slow, so build a
# === lib to speed up
# ========================================
set(COMMON_SOURCES
${PROJECT_SOURCE_DIR}/Examples/RGB-D/ty/common/MatViewer.cpp
${PROJECT_SOURCE_DIR}/Examples/RGB-D/ty/common/PointCloudViewer.cpp
    )
add_library(sample_common STATIC ${COMMON_SOURCES}) #=====================

# Build examples
# rgb-d camera example
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
#add_executable(rgbd_tum
#Examples/RGB-D/rgbd_tum.cc)
#target_link_libraries(rgbd_tum ${PROJECT_NAME})

# rgb-d camera example
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
add_executable(ty_rgbd Examples/RGB-D/my_rgbd.cc)
target_link_libraries(ty_rgbd ${PROJECT_NAME} sample_common ${TY_TARGET_LIB})# Two more link libraries

# Example of a binocular camera
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
#add_executable(stereo_kitti #KITTI data set
##Examples/Stereo/stereo_kitti.cc)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})
#add_executable(stereo_euroc # EuRoC data set
#Examples/Stereo/stereo_euroc.cc)
#target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(my_stereo # stereo_on_line
Examples/Stereo/my_stereo.cc)
target_link_libraries(my_stereo ${PROJECT_NAME})


#Monocular camera example
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
#add_executable(mono_tum # TUM data set
#Examples/Monocular/mono_tum.cc)
#target_link_libraries(mono_tum ${PROJECT_NAME})
#add_executable(mono_kitti #KITTI data set
#Examples/Monocular/mono_kitti.cc)
#target_link_libraries(mono_kitti ${PROJECT_NAME})
#add_executable(mono_euroc # EuRoC data set
#Examples/Monocular/mono_euroc.cc)
#target_link_libraries(mono_euroc ${PROJECT_NAME})

## .txt >>> .bin file conversion
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
#add_executable(
#bin_vocabulary
#tools/bin_vocabulary.cc)
#target_link_libraries(bin_vocabulary ${PROJECT_NAME})

