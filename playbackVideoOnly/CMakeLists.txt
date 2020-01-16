cmake_minimum_required (VERSION 2.6)
project (SonarVision)

find_package (OpenCV 3.0.0 EXACT REQUIRED )

# fetch all the local directories for generated code
get_filename_component(SONAR_VISION_BIN_DIR ./bin ABSOLUTE)
get_filename_component(SONAR_VISION_INC_DIR ./include ABSOLUTE)
get_filename_component(SONAR_VISION_BUILD_DIR ./build ABSOLUTE)
get_filename_component(SONAR_API /home/anthony/sonar-processing/bvtsdk ABSOLUTE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall") # -fprofile-arcs -f$
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall") # -fprofile-arcs -ftest$

set(EXECUTABLE_OUTPUT_PATH ${SONAR_VISION_BIN_DIR})         

#include_directories(${OpenCV_INCLUDE_DIRS} /home/syllogismrxs/builds/image-processing/bvtsdk/include)
include_directories(${OpenCV_INCLUDE_DIRS} ${SONAR_API}/include)

add_executable (sonar_opencv opencv.cpp)

#target_link_libraries(sonar_opencv ${OpenCV_LIBS} /home/syllogismrxs/builds/image-processing/bvtsdk/lib/libbvtsdk.so)
target_link_libraries(sonar_opencv ${OpenCV_LIBS} ${SONAR_API}/lib/libbvtsdk.so)

