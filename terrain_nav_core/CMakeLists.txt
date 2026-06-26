cmake_minimum_required(VERSION 3.18)
project(terrain_nav_core LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _terrain_nav_core
    bindings.cpp
    correlation_core.cpp
)

target_compile_features(_terrain_nav_core PRIVATE cxx_std_17)
set_target_properties(_terrain_nav_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
