cmake_minimum_required(VERSION 3.16)
project(neat_native_development_prerequisites LANGUAGES CXX)

# Compile only. This validates the selected native compiler's public header
# closure without running a probe or adding SDK-specific include directories.
add_library(native_development_prerequisites OBJECT headers.cpp)
target_compile_features(native_development_prerequisites PRIVATE cxx_std_20)
