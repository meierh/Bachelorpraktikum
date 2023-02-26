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

void compareAreaConnecMap(const AreaConnectivity::AreaConnecMap& mapPar,const AreaConnectivity::AreaConnecMap& mapSeq);

void compareEdgeLengthHistogram(const Histogram::HistogramData& histogramPar, const Histogram::HistogramData& histogramSeq, const double epsilon);
