#pragma once

#include "Distance.h"
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
#include <tuple>
#include <utility>

#include <iostream>
#include <sstream>

class AllPairsShortestPath {
	static std::tuple<double, double, std::uint64_t> compute_sssp(const DistributedGraph& graph, unsigned int node_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution) {
		const auto my_rank = MPIWrapper::get_my_rank();

		std::ofstream log_file{ std::string("log_apsp_") + std::to_string(my_rank) + ".log" };

		std::vector<double> distances(total_number_nodes, std::numeric_limits<double>::infinity());
		std::priority_queue<VertexDistance, std::vector<VertexDistance>, std::greater<VertexDistance>> shortest_paths_queue{};

		const auto root_id = prefix_distribution[my_rank] + node_id;

		distances[root_id] = 0;
		shortest_paths_queue.emplace(my_rank, node_id, 0);

		while (!shortest_paths_queue.empty()) {
			const auto [current_rank, current_id, current_distance] = shortest_paths_queue.top();
			shortest_paths_queue.pop();

			const auto out_edges = graph.get_out_edges(current_rank, current_id);

			for (const auto& [target_rank, target_id, weight] : out_edges) {
				const auto new_distance = current_distance + std::abs(weight);

				const auto other_node_id = prefix_distribution[target_rank] + target_id;

				if (distances[other_node_id] <= new_distance) {
					continue;
				}

				distances[other_node_id] = new_distance;
				shortest_paths_queue.emplace(target_rank, target_id, new_distance);
			}
		}

		auto sum_efficiency = 0.0;
		auto sum_shortest_path_from_node = 0.0;
		auto number_unreachables_from_node = std::uint64_t(0);

		for (auto i = 0; i < distances.size(); i++) {
			const auto distance = distances[i];

			if (distance == std::numeric_limits<double>::infinity()) {
				number_unreachables_from_node++;
				log_file << node_id << '\t' << i << '\n';
			}
			else {
				sum_shortest_path_from_node += distance;
				if (distance != 0) {
					sum_efficiency += (1.0 / distance);
				}
			}
		}

		return { sum_shortest_path_from_node, sum_efficiency, number_unreachables_from_node };
	}

public:
	static std::tuple<double, double, std::uint64_t> compute_apsp(const DistributedGraph& graph) {
		const auto my_rank = MPIWrapper::get_my_rank();

		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

		const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);
		const auto prefix_distribution = calculate_prefix_sum(node_distribution);

		auto status_counter = 0;
		const auto status_step = number_local_nodes / 101.0;

		auto sum_shortest_path_locally = 0.0;
		auto sum_efficiency_locally = 0.0;
		auto number_unreachables_locally = std::uint64_t(0);

		graph.lock_all_rma_windows();

		for (auto node_id = 0U; node_id < number_local_nodes; node_id++) {
			const auto [sum_shortest_path_from_node, sum_efficiency_from_node, number_unreachables_from_node]
				= compute_sssp(graph, node_id, total_number_nodes, prefix_distribution);

			sum_shortest_path_locally += sum_shortest_path_from_node;
			sum_efficiency_locally += sum_efficiency_from_node;
			number_unreachables_locally += number_unreachables_from_node;

			if (node_id % static_cast<int>(status_step) == 0 && status_counter < 100) {
				status_counter++;
				Status::report_status(node_id, total_number_nodes, "APSP");
			}
		}

		MPIWrapper::barrier();

		graph.unlock_all_rma_windows();

		Status::report_status(number_local_nodes, total_number_nodes, "APSP");

		const auto sum_shortest_paths = MPIWrapper::reduce_sum(sum_shortest_path_locally);
		const auto sum_efficiency = MPIWrapper::reduce_sum(sum_efficiency_locally);
		const auto sum_unreachables = MPIWrapper::reduce_sum(number_unreachables_locally);

		const auto number_pairs_without_self = (total_number_nodes * total_number_nodes) - total_number_nodes;
		const auto number_reached_pairs = number_pairs_without_self - number_unreachables_locally;

		const auto average_shortest_path_length = sum_shortest_paths / number_reached_pairs;
		const auto average_efficiency = sum_efficiency / number_reached_pairs;

		return { average_shortest_path_length, average_efficiency, sum_unreachables };
	}
};
