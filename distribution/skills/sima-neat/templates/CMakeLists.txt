cmake_minimum_required(VERSION 3.16)
project(neat_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)

add_executable(neat_app main.cpp)
target_link_libraries(neat_app PRIVATE SimaNeat::sima_neat)
