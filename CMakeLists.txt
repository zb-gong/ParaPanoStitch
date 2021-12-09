cmake_minimum_required(VERSION 3.11)
project(pps)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


# include_directories(util)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)
set(PSS_SOURCE ${DIR_SRCS})

add_executable(${PROJECT_NAME} ${PSS_SOURCE})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
