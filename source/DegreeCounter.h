#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include <algorithm>
#include <iostream>
#include <limits>

struct MinMax {
	std::uint64_t min;
	std::uint64_t max;
};

class InDegreeCounter {
public:
	static MinMax count_in_degrees(const DistributedGraph& graph) {
		const auto my_rank = MPIWrapper::get_my_rank();
		const auto number_local_nodes = graph.get_number_local_nodes();

		auto current_min = std::numeric_limits<std::uint64_t>::max();
		auto current_max = std::numeric_limits<std::uint64_t>::min();

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto in_degree = graph.get_number_in_edges(my_rank, node_id);

			current_min = std::min(current_min, in_degree);
			current_max = std::max(current_max, in_degree);
		}

		const auto total_min = MPIWrapper::reduce_min(current_min);
		const auto total_max = MPIWrapper::reduce_max(current_max);

		return { total_min, total_max };
	}
};

class OutDegreeCounter {
public:
	static MinMax count_out_degrees(const DistributedGraph& graph) {
		const auto my_rank = MPIWrapper::get_my_rank();
		const auto number_local_nodes = graph.get_number_local_nodes();

		auto current_min = std::numeric_limits<std::uint64_t>::max();
		auto current_max = std::numeric_limits<std::uint64_t>::min();

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto in_degree = graph.get_number_out_edges(my_rank, node_id);

			current_min = std::min(current_min, in_degree);
			current_max = std::max(current_max, in_degree);
		}

		const auto total_min = MPIWrapper::reduce_min(current_min);
		const auto total_max = MPIWrapper::reduce_max(current_max);

		return { total_min, total_max };
	}
};
