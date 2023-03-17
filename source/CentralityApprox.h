#pragma once

#include "CommunicationPatterns.h"
#include "DistributedGraph.h"
#include "EdgeLength.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include "Status.h"
#include "Util.h"

#include <cassert>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

class BetweennessCentralityApproximation {

/*
* The Implementation is based on the Framework presented in the Paper:
  "Bavarian: Betweenness Centrality Approximation with Variance-Aware Rademacher Averages"
*/

public:
	struct BC_hash {
		template <class T1, class T2>
		std::size_t operator()(const std::pair<T1, T2>& p) const {
			// Szudzik's function: a >= b ? a * a + a + b : a + b * b; where a, b >= 0 ()
			// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
			return (p.first >= p.second) ? (p.first * p.first + p.first + p.second)
						     : (p.first + p.second * p.second);
		}
	};

	using BC_e = std::pair<std::unordered_map<std::pair<int, int>, double, BC_hash>, double>;

	static std::unique_ptr<BC_e> compute_betweenness_centrality_approx(const DistributedGraph& graph, int m,
									   double d, int k,
									   unsigned int result_rank = 0);

	// private:
	/*
	The helper functions use int values as unique node ids while BC returns nodes as (rank_id, node_id) pairs.
	*/
	static std::pair<int, int> draw_sample(const DistributedGraph& graph, int number_ranks, int number_local_nodes,
					       std::vector<std::uint64_t> prefix_distribution);

	static std::unordered_map<int, double>
	get_function_values(const DistributedGraph& graph, std::pair<int, int> sample,
			    const std::vector<std::uint64_t>& prefix_distribution, const uint64_t total_number_nodes);

	static std::vector<double> draw_rademacher(int k);

	static std::vector<std::vector<NodePath>> compute_sssp(const DistributedGraph& graph, unsigned int src_id,
							       unsigned int dest_id, std::uint64_t total_number_nodes,
							       const std::vector<std::uint64_t>& prefix_distribution);

	static double get_epsilon(std::vector<std::vector<double>> sums, int m, double d, int k);

};
