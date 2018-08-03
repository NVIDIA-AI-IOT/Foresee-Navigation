cmake_minimum_required(VERSION 3.5)
project(SXLT)

find_package(OpenCV REQUIRED) # after project()


set(CMAKE_CXX_STANDARD 11)

add_executable(SXLT main.cpp)
#add_executable(SXLT_Viewer.cpp)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(vendor)


target_link_libraries(SXLT ${OpenCV_LIBS})
#target_link_libraries(SXLT_Viewer ${OpenCV_LIBS})

configure_file(SXLT.ini SXLT.ini COPYONLY)
