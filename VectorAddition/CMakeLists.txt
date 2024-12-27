cmake_minimum_required (VERSION 3.29)

project ("VectorAddition" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(UTILS_DIR "../utils")
file(GLOB UTIL_HEADERS "${UTILS_DIR}/*.hpp")

include_directories("../utils")

add_executable (VectorAddition "main.cpp")
