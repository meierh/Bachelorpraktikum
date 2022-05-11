#pragma once

#include "DistributedGraph.h"
#include "EdgeCounter.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"

#include <cstdint>
#include <utility>
#include <vector>

class EdgeLength {
	static double compute_edge_length(const DistributedGraph& graph, const int node_id) {
		const auto my_rank = MPIWrapper::get_my_rank();
	
		auto accumulated_distance = 0.0;
		const auto& node_position = graph.get_node_position(my_rank, node_id);
		const auto& out_edges = graph.get_out_edges(my_rank, node_id);

		for (const auto& [target_rank, target_id, weight] : out_edges) {
			const auto& position = graph.get_node_position(target_rank, target_id);

			const auto& difference = position - node_position;
			const auto& distance = difference.calculate_2_norm();

			accumulated_distance += distance;
		}

		return accumulated_distance;
	}

public:
	static double compute_edge_length(const DistributedGraph& graph) {
		const auto number_local_nodes = graph.get_number_local_nodes();

		auto accumulated_distance = 0.0;

		graph.lock_all_rma_windows();

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto& distance = compute_edge_length(graph, node_id);
			accumulated_distance += distance;
		}

		graph.unlock_all_rma_windows();

		const auto my_rank = MPIWrapper::get_my_rank();
		const auto total_distance = MPIWrapper::reduce_sum(accumulated_distance);
		const auto number_total_edges = OutEdgeCounter::count_out_edges(graph);

		if (my_rank != 0) {
			return 0.0;
		}

		const auto& average_distance = total_distance / number_total_edges;

		return average_distance;
	}

	static double all_compute_edge_length(const DistributedGraph& graph) {
		const auto number_local_nodes = graph.get_number_local_nodes();

		auto accumulated_distance = 0.0;

		graph.lock_all_rma_windows();

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto& distance = compute_edge_length(graph, node_id);
			accumulated_distance += distance;
		}

		graph.unlock_all_rma_windows();

		const auto my_rank = MPIWrapper::get_my_rank();
		const auto total_distance = MPIWrapper::all_reduce_sum(accumulated_distance);
		const auto number_total_edges = OutEdgeCounter::all_count_out_edges(graph);

		const auto& average_distance = total_distance / number_total_edges;

		return average_distance;
	}
};
