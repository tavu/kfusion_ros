cmake_minimum_required(VERSION 2.8.3)
project(kfusion_ros)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  image_transport
  tf2
)

catkin_package(
CATKIN_DEPENDS
)

find_package(TooN   REQUIRED)
find_package(Eigen3   REQUIRED)

#set(TOON_INCLUDE_PATHS "~/src/TooN")
#set(EIGEN3_INCLUDE_DIR "~/src/eigen")




find_package(CUDA)
SET( CUDA_PROPAGATE_HOST_FLAGS OFF ) 
set(kfusion_cuda_srcs src/kernels.cu)
    cuda_add_library(kfusion_cuda    SHARED  ${kfusion_cuda_srcs}  OPTIONS -O3 -std=c++11 -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES 
    -I${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_headers 
    -I${CMAKE_CURRENT_SOURCE_DIR}/src 
    -I${TOON_INCLUDE_PATHS}
    -I${EIGEN3_INCLUDE_DIR}
    )

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TOON_INCLUDE_PATHS}
  ${EIGEN3_INCLUDE_DIR}
  ./src
  ./src/cuda_headers
)

add_executable( kfusion_node 
    src/kfusion_node.cpp 
)



target_link_libraries(kfusion_node
    ${catkin_LIBRARIES}  
    kfusion_cuda
)

