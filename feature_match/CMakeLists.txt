cmake_minimum_required(VERSION 3.1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project( FeatureDetect )

find_package( OpenCV REQUIRED )
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable( FeatureDetect SURF_detection_Demo.cpp )
target_link_libraries( FeatureDetect ${OpenCV_LIBS} )

project( FeatureMatch )

find_package( OpenCV REQUIRED )

add_executable( FeatureMatch SURF_FLANN_matching_Demo.cpp )
target_link_libraries( FeatureMatch ${OpenCV_LIBS} )
