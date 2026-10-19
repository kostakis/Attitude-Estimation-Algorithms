cmake_minimum_required(VERSION 3.10)
project(AttitudeEstimation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find Eigen3 (required dependency)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Set output directories to match Makefile structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Collect all source files from src/ directory
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Create object library for shared source files
# This avoids recompiling common sources for each executable
add_library(common_sources OBJECT ${SOURCES})

# Complementary Filter executable
add_executable(complmentary.out
    ${PROJECT_SOURCE_DIR}/complentaryFilterMain.cpp
    $<TARGET_OBJECTS:common_sources>
)

# EKF Filter executable
add_executable(ekf.out
    ${PROJECT_SOURCE_DIR}/ekfFilterMain.cpp
    $<TARGET_OBJECTS:common_sources>
)

# Mahony Filter executable
add_executable(mahony.out
    ${PROJECT_SOURCE_DIR}/mahonyFilterMain.cpp
    $<TARGET_OBJECTS:common_sources>
)

# Custom targets to match Makefile target names
add_custom_target(complemntaryFilter DEPENDS complmentary.out)
add_custom_target(ekfFilter DEPENDS ekf.out)
add_custom_target(mahonyFilter DEPENDS mahony.out)

# Build all filters (matches Makefile's 'all' target)
add_custom_target(all_filters DEPENDS complmentary.out ekf.out mahony.out)

# Custom clean target to remove build and binary directories
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove *.out
    COMMENT "Cleaning build artifacts, binaries, and *.out files"
)

# Run targets for convenience (matches Makefile pattern)
add_custom_target(run_complementary
    COMMAND ${PROJECT_SOURCE_DIR}/bin/complmentary.out
    DEPENDS complmentary.out
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running Complementary Filter"
)

add_custom_target(run_ekf
    COMMAND ${PROJECT_SOURCE_DIR}/bin/ekf.out
    DEPENDS ekf.out
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running EKF Filter"
)

add_custom_target(run_mahony
    COMMAND ${PROJECT_SOURCE_DIR}/bin/mahony.out
    DEPENDS mahony.out
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running Mahony Filter"
)

# Run all filters
add_custom_target(run_all
    DEPENDS run_complementary run_ekf run_mahony
    COMMENT "Running all filters"
)
