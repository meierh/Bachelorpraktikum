#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include <iostream>

class InEdgeCounter {
public:
	static std::uint64_t count_in_edges(const DistributedGraph& graph) {
		const auto number_local_in_edges = graph.get_number_local_in_edges();
		const auto number_total_in_edges = MPIWrapper::reduce_sum(number_local_in_edges);

		return number_total_in_edges;
	}

	static std::uint64_t all_count_in_edges(const DistributedGraph& graph) {
		const auto number_local_in_edges = graph.get_number_local_in_edges();
		const auto number_total_in_edges = MPIWrapper::all_reduce_sum(number_local_in_edges);

		return number_total_in_edges;
	}
};

class OutEdgeCounter {
public:
	static std::uint64_t count_out_edges(const DistributedGraph& graph) {
		const auto number_local_out_edges = graph.get_number_local_out_edges();
		const auto number_total_out_edges = MPIWrapper::reduce_sum(number_local_out_edges);

		return number_total_out_edges;
	}

	static std::uint64_t all_count_out_edges(const DistributedGraph& graph) {
		const auto number_local_out_edges = graph.get_number_local_out_edges();
		const auto number_total_out_edges = MPIWrapper::all_reduce_sum(number_local_out_edges);

		return number_total_out_edges;
	}
};
