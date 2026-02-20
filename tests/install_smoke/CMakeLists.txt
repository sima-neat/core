cmake_minimum_required(VERSION 3.16)
project(SimaNeatInstallSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SimaNeat REQUIRED)

add_executable(install_smoke_app main.cpp)
target_link_libraries(install_smoke_app PRIVATE SimaNeat::sima_neat)
