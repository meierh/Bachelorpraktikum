#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "Centrality.h"
#include "CentralityApprox.h"
#include "Clustering.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

#include "AreaConnectivity.h"
#include "Histogram.h"
#include "Modularity.h"

void test_algorithm_parallelization(std::filesystem::path input_directory);

void compare_area_connec_map(const AreaConnectivity::AreaConnecMap& map_par,const AreaConnectivity::AreaConnecMap& map_seq);

void compare_edge_length_histogram(const Histogram::HistogramData& histogram_par, const Histogram::HistogramData& histogram_seq, const double epsilon);
