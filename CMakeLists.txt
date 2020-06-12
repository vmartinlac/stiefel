
cmake_minimum_required(VERSION 3.0)

find_package(Eigen3 REQUIRED)

add_executable(stiefel main.cpp Stiefel.h Stiefel.cpp)
set_target_properties(stiefel PROPERTIES CXX_STANDARD 11)
#target_link_libraries(stiefel PUBLIC Eigen3::Eigen)
target_include_directories(stiefel PUBLIC ${EIGEN3_INCLUDE_DIR})

