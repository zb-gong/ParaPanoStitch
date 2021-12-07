cmake_minimum_required(VERSION 3.15)
project(pps)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA 9.0 REQUIRED)

# include_directories(util)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

aux_source_directory(src DIR_SRCS)
set(PSS_SOURCE ${DIR_SRCS})

cuda_add_executable(${PROJECT_NAME} ${PSS_SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
