cmake_minimum_required(VERSION 2.8)
project( CutVideo )
find_package( OpenCV 3.0.0 EXACT REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( CutVideo cut_video.cpp )
target_link_libraries( CutVideo ${OpenCV_LIBS} )
