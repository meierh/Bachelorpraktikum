#pragma once

#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include "Status.h"
#include "Util.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <queue>
#include <utility>

#include <iostream>
#include <sstream>

struct VertexDistanceTriple {
	int mpi_rank;
	int node_id;
	unsigned int distance;
};

template <>
struct std::greater<VertexDistanceTriple> {
	bool operator()(const VertexDistanceTriple& lhs, const VertexDistanceTriple& rhs) const {
		return lhs.distance > rhs.distance;
	}
};

class AllPairsShortestPath {
	static std::pair<double, std::uint64_t> compute_sssp(const DistributedGraph& graph, int node_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution) {
		const auto my_rank = MPIWrapper::get_my_rank();

		std::ofstream of{ std::string("log_") + std::to_string(my_rank), std::ios_base::app };

		std::vector<double> distances(total_number_nodes, std::numeric_limits<double>::infinity());
		std::priority_queue<VertexDistanceTriple, std::vector<VertexDistanceTriple>, std::greater<VertexDistanceTriple>> shortest_paths_queue{};

		const auto root_id = prefix_distribution[my_rank] + node_id;

		distances[root_id] = 0;
		shortest_paths_queue.emplace(my_rank, node_id, 0);

		while (!shortest_paths_queue.empty()) {
			const auto [current_rank, current_id, current_distance] = shortest_paths_queue.top();
			shortest_paths_queue.pop();

			const auto out_edges = graph.get_out_edges(current_rank, current_id);

			for (const auto& out_edge : out_edges) {
				const auto new_distance = current_distance + std::abs(out_edge.weight);

				const auto other_node_id = prefix_distribution[out_edge.target_rank] + out_edge.target_id;
				if (distances[other_node_id] > new_distance) {
					distances[other_node_id] = new_distance;
					shortest_paths_queue.emplace(out_edge.target_rank, out_edge.target_id, new_distance);
				}
			}
		}

		auto sum = 0.0;
		std::uint64_t inf_counter = 0;

		for (auto i = 0; i < distances.size(); i++) {
			const auto distance = distances[i];

			if (distance == std::numeric_limits<double>::infinity()) {
				inf_counter++;
				of << node_id << '\t' << i << '\n';
			}
			else {
				sum += distance;
			}
		}

		return { sum, inf_counter };
	}

public:
	static std::pair<double, std::uint64_t> compute_apsp(const DistributedGraph& graph) {		
		const auto my_rank = MPIWrapper::get_my_rank();

		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

		const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);
		const auto prefix_distribution = calculate_prefix_sum(node_distribution);

		auto status_counter = 0;
		const auto status_step = number_local_nodes / 101.0;

		double sum = 0.0;
		std::uint64_t inf_counter = 0;

		graph.lock_all_rma_windows();

		for (auto node_id = 0; node_id < number_local_nodes; node_id++) {
			const auto [sssp, infs] = compute_sssp(graph, node_id, total_number_nodes, prefix_distribution);
			sum  += sssp;
			inf_counter += infs;

			if (node_id % static_cast<int>(status_step) == 0 && status_counter < 100) {
				status_counter++;
				Status::report_status(node_id, total_number_nodes, "APSP");
			}
		}

		MPIWrapper::barrier();

		graph.unlock_all_rma_windows();

		Status::report_status(number_local_nodes, total_number_nodes, "APSP");

		const auto total_sum = MPIWrapper::reduce_sum(sum);

		const auto total_infs = MPIWrapper::reduce_sum(inf_counter);

		return { total_sum / ((total_number_nodes * total_number_nodes) - inf_counter), total_infs};
	}
};
