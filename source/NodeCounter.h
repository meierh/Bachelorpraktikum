#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include <cstdint>
#include <vector>

class NodeCounter {
public:
	static std::uint64_t count_nodes(const DistributedGraph& graph) {
		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto number_total_nodes = MPIWrapper::reduce_sum(number_local_nodes);

		return number_total_nodes;
	}

	static std::uint64_t all_count_nodes(const DistributedGraph& graph) {
		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto number_total_nodes = MPIWrapper::all_reduce_sum(number_local_nodes);

		return number_total_nodes;
	}
};

class NodeDistributionCounter {
public:
	static std::vector<std::uint64_t> all_count_node_distribution(const DistributedGraph& graph) {
		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto node_distribution = MPIWrapper::all_gather(number_local_nodes);

		return node_distribution;
	}
};
