#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "Centrality.h"
#include "Clustering.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

#include "AreaConnectivity.h"
#include "CentralityApprox.h"
#include "EdgeCounter.h"
#include "Histogram.h"
#include "Modularity.h"
#include "NetworkMotifs.h"

class AlgorithmTests {
public:
	static void test_algorithm_parallelization(std::filesystem::path input_directory);

	static void test_centrality_approx(std::filesystem::path input_directory);

	static void check_graph_property(std::filesystem::path input_directory);

	static void compare_area_connec_map(const AreaConnectivity::AreaConnecMap& map_par, const AreaConnectivity::AreaConnecMap& map_seq);

	static void compare_edge_length_histogram(const Histogram::HistogramData& histogram_par, const Histogram::HistogramData& histogram_seq,
						  const double epsilon);

	struct StdPair_hash {
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
		}
	};

	struct StdDoublePair_hash {
		StdPair_hash hash;
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			return hash(p.first) ^ hash(p.second);
		}
	};
};