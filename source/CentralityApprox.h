#pragma once

// Evtl. unnötige imports enthalten
#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include "Status.h"
#include "DistributedGraph.h"
#include "EdgeLength.h"
#include "Util.h"
#include "CommunicationPatterns.h"

#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <vector>
#include <tuple>
#include <cassert>  // debug
#include <random>
#include <math.h>


class BetweennessCentralityApproximation {
	
public:

	/*
	struct BC_hash
    {
        std::size_t operator () (const std::pair<int, int> &p) const 
        {
			// Szudzik's function: a >= b ? a * a + a + b : a + b * b; where a, b >= 0 ()
			// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
			// (besser als Cantor Pairing)
			return (p.first >= p.second) ? (p.first * p.first + p.first + p.second) 
										 : (p.first + p.second * p.second);
        }
    };
	*/
	struct BC_hash
    {
        template <class T1, class T2>
        std::size_t operator () (const std::pair<T1,T2> &p) const 
        {
            // Szudzik's function: a >= b ? a * a + a + b : a + b * b; where a, b >= 0 ()
			// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
			// (besser als Cantor Pairing)
			return (p.first >= p.second) ? (p.first * p.first + p.first + p.second) 
										 : (p.first + p.second * p.second);
        }
    };


	//using Node = std::pair<int, int>;	// Node = (rank_id, node_id)
	//using BC = std::unordered_map<Node, double, BC_hash>; // needs explicit hash function
	using BC_e = std::pair<std::unordered_map<std::pair<int, int>, double, BC_hash>, double>;
	
	static std::unique_ptr<BC_e> compute_betweenness_centrality_approx(const DistributedGraph& graph, int m, double d, int k, unsigned int result_rank=0);

private:
	/*
	The helper functions use int values as unique node ids while BC returns nodes as (rank_id, node_id) pairs.
	*/
	static std::pair<int, int> drawSample(const DistributedGraph& graph, int number_ranks, int number_local_nodes, std::vector<std::uint64_t> prefix_distribution);
	static std::unordered_map<int, double> getFunctionValues(const DistributedGraph& graph, std::pair<int, int> sample, const std::vector<std::uint64_t>& prefix_distribution);
	static std::vector<double> drawRademacher(int k);
	static std::vector<std::vector<NodePath>> compute_sssp(const DistributedGraph& graph, unsigned int node_id, unsigned int dest_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution);

	static double getEpsilon(std::vector<std::vector<double>> sums, int m, double d, int k);

	//static convertNodesToUniques();
	//static convertUniquesToNodes();

};
