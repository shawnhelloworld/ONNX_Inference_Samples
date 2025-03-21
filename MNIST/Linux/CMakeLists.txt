cmake_minimum_required(VERSION 3.10)
project(mnist CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(ONNXRUNTIME_INCLUDE_DIR "/usr/local/onnxruntime/include")
set(ONNXRUNTIME_LIB_DIR "/usr/local/onnxruntime/lib")

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
add_executable(mnist MNIST.cpp)
target_link_libraries(mnist ${SDL2_LIBRARIES} onnxruntime_providers_shared onnxruntime)
