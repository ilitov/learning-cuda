cmake_minimum_required (VERSION 3.29)

project ("VectorAddition" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -v")

set(UTILS_DIR "../utils")
file(GLOB UTIL_HEADERS "${UTILS_DIR}/*.hpp")

include_directories("../utils")

add_executable (${PROJECT_NAME} ${UTIL_HEADERS} "main.cpp" "kernel.hpp" "kernel.cu")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE cudart_static)
