cmake_minimum_required(VERSION 3.20)
project(hw1_simd CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(hwy REQUIRED)
find_package(TBB REQUIRED)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.5
)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

FetchContent_MakeAvailable(googlebenchmark googletest)

add_library(ops ops.cpp)
target_include_directories(ops PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(ops PUBLIC hwy::hwy TBB::tbb)

add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE ops GTest::gtest_main)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE ops benchmark::benchmark benchmark::benchmark_main)