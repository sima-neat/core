cmake_minimum_required(VERSION 3.20)
project(transfer_cache_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)
add_executable(transfer_bench transfer_bench.cpp)
target_link_libraries(transfer_bench PRIVATE SimaNeat::sima_neat)
