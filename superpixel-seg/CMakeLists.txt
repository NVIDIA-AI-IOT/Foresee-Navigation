cmake_minimum_required(VERSION 3.5)
project(superpixel_seg)

find_package(OpenCV REQUIRED) # after project()

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_CXX_FLAGS -fpermissive)
#set(CMAKE_CXX_FLAGS -Wall)


file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.hpp )
add_library( local_lib ${LIB_SOURCES} ${LIB_HEADERS})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(local_lib ${OpenCV_LIBS})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS graph)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(local_lib ${Boost_LIBRARIES})
endif()




add_executable( superpixel_viewer exe/superpixel_seg.cpp )
target_link_libraries( superpixel_viewer local_lib )

add_executable( superpixel_evaluate exe/superpixel_evaluate.cpp )
target_link_libraries( superpixel_evaluate local_lib )

add_executable( perspective_finder exe/perspective_finder.cpp exe/train_forest.cpp exe/superpixel_streamer.cpp)
target_link_libraries( perspective_finder local_lib )

add_executable( train_forest exe/train_forest.cpp )
target_link_libraries( train_forest local_lib )

add_executable( superpixel_streamer exe/superpixel_streamer.cpp )
target_link_libraries( superpixel_streamer local_lib )


configure_file(perspective_calibrate.png perspective_calibrate.png COPYONLY)
