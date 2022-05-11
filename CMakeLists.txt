cmake_minimum_required (VERSION 3.8)

project ("ScalableGraphAlgorithm")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_subdirectory ("source")
