cmake_minimum_required(VERSION 3.15)
project(pps)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

# include_directories(util)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)
set(PSS_SOURCE ${DIR_SRCS})

add_executable(${PROJECT_NAME} ${PSS_SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
