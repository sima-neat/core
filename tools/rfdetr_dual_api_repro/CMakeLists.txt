cmake_minimum_required(VERSION 3.16)
project(rfdetr_dual_api_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(rfdetr_cpp_repro rfdetr_cpp_repro.cpp)
target_link_libraries(rfdetr_cpp_repro PRIVATE SimaNeat::sima_neat)
