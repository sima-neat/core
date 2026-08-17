cmake_minimum_required(VERSION 3.16)

project(SimaPCIeHostPackageSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaPCIeHost REQUIRED CONFIG)

add_executable(sima_pcie_host_package_smoke main.cpp)
target_link_libraries(
  sima_pcie_host_package_smoke
  PRIVATE SimaPCIeHost::sima_neat_pcie_host
)
