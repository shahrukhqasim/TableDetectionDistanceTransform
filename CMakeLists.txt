cmake_minimum_required(VERSION 3.6)
project(preprocessing)

set(CMAKE_CXX_STANDARD 11)
link_directories(/usr/local/lib/ /usr/lib/)
include_directories(/usr/local/include/)


set(SOURCE_FILES main.cpp)
add_executable(preprocessing ${SOURCE_FILES})

target_link_libraries(preprocessing opencv_core opencv_imgproc tesseract  lept opencv_highgui opencv_imgcodecs)