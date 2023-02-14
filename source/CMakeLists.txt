cmake_minimum_required (VERSION 3.8)
include(FetchContent)
set (CMAKE_CXX_STANDARD 20)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        815553211b644b2a64b5dc671c602233c49c065f
)
FetchContent_MakeAvailable(CLI11)

add_executable(ScalableGraphAlgorithm "ScalableGraphAlgorithm.cpp" "DistributedGraph.cpp" "CommunicationPatterns.cpp" "AreaConnectivity.cpp" "Histogram.cpp" "Modularity.cpp" "NetworkMotifs.cpp")

target_include_directories(ScalableGraphAlgorithm SYSTEM PRIVATE ${CLI11_SOURCE_DIR}/include)
target_link_libraries(ScalableGraphAlgorithm PRIVATE ${MPI_LIBRARIES})
