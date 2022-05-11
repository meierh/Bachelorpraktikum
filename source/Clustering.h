#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include "Status.h"

#include <cstdint>
#include <utility>
#include <vector>

class Clustering {
	static std::uint64_t count_matches(const std::vector<InEdge>& in_edges, const std::vector<OutEdge>& out_edges) {
		auto counter = std::uint64_t(0);

		for (const auto& [rank_1, id_1, weight_1] : in_edges) {
			for (const auto& [rank_2, id_2, weight_2] : out_edges) {
				if (rank_1 == rank_2 && id_1 == id_2) {
					counter++;
					break;
				}
			}
		}

		return counter;
	}

	static double compute_acc_node(const DistributedGraph& graph, int node_id) {
		const auto my_rank = MPIWrapper::get_my_rank();
		
		auto number_closed_triangles = std::uint64_t(0);

		const auto in_edges = graph.get_in_edges(my_rank, node_id);
		const auto out_edges = graph.get_out_edges(my_rank, node_id);

		const auto number_all_trianles = static_cast<std::uint64_t>(in_edges.size() * out_edges.size());

		for (const auto& [target_rank, target_id, weight] : out_edges) {
			const auto other_out_edges = graph.get_out_edges(target_rank, target_id);
			
			const auto number_matches = count_matches(in_edges, other_out_edges);
			number_closed_triangles += number_matches;
		}

		return static_cast<double>(number_closed_triangles) / static_cast<double>(number_all_trianles);
	}

public:
	static double compuate_average_clustering_coefficient(const DistributedGraph& graph) {
		const auto my_rank = MPIWrapper::get_my_rank();

		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

		auto status_counter = 0;
		const auto status_step = number_local_nodes / 101.0;

		graph.lock_all_rma_windows();

		auto sum_clustering_coefficient_locally = 0.0;

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto clustering_coefficient_node = compute_acc_node(graph, node_id);

			sum_clustering_coefficient_locally += clustering_coefficient_node;

			if (node_id % static_cast<int>(status_step) == 0 && status_counter < 100) {
				status_counter++;
				Status::report_status(node_id, total_number_nodes, "Clustering Coefficient");
			}
		}

		MPIWrapper::barrier();

		graph.unlock_all_rma_windows();

		Status::report_status(number_local_nodes, total_number_nodes, "Clustering Coefficient");

		const auto sum_clustering_coefficient = MPIWrapper::reduce_sum(sum_clustering_coefficient_locally);
		const auto average_clustering_coefficient = sum_clustering_coefficient / total_number_nodes;
		return average_clustering_coefficient;
	}
};
