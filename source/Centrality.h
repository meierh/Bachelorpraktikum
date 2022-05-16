#pragma once

#include "Distance.h"
#include "DistributedGraph.h"
#include "MPIWrapper.h"
#include "NodeCounter.h"
#include "Status.h"
#include "Util.h"

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

class BetweennessCentrality {
	static void compute_betweenness_centrality(const DistributedGraph& graph, unsigned int node_id, std::uint64_t total_number_nodes, const std::vector<std::uint64_t>& prefix_distribution, std::vector<double>& local_betweenness) {
		const auto my_rank = MPIWrapper::get_my_rank();

		std::vector<std::vector<NodePath>> shortest_paths(total_number_nodes, std::vector<NodePath>{});
		std::vector<double> distances(total_number_nodes, std::numeric_limits<double>::infinity());

		std::priority_queue<VertexDistancePath, std::vector<VertexDistancePath>, std::greater<VertexDistancePath>> shortest_paths_queue{};

		const auto root_id = prefix_distribution[my_rank] + node_id;

		distances[root_id] = 0;

		NodePath start(20);
		start.append_node(node_id);

		shortest_paths[root_id] = { start };
		shortest_paths_queue.emplace(my_rank, node_id, 0, std::move(start));

		while (!shortest_paths_queue.empty()) {
			const auto [current_rank, current_id, current_distance, current_path] = shortest_paths_queue.top();
			shortest_paths_queue.pop();

			const auto& out_edges = graph.get_out_edges(current_rank, current_id);

			for (const auto& [target_rank, target_id, weight] : out_edges) {
				const auto new_distance = current_distance + std::abs(weight);

				const auto other_node_id = prefix_distribution[target_rank] + target_id;
				if (distances[other_node_id] < new_distance) {
					continue;
				}

				if (distances[other_node_id] > new_distance) {
					distances[other_node_id] = new_distance;
					shortest_paths[other_node_id].clear();
				}

				NodePath new_path = current_path;
				new_path.append_node(other_node_id);

				shortest_paths_queue.emplace(target_rank, target_id, new_distance, new_path);
				shortest_paths[other_node_id].emplace_back(std::move(new_path));
			}
		}

		for (const auto& shortest_paths_to_node : shortest_paths) {
			const auto number_shortest_paths = shortest_paths_to_node.size();
			const auto portion = 1.0 / number_shortest_paths;

			for (const auto& shortest_path_to_node : shortest_paths_to_node) {
				const auto& nodes_on_path = shortest_path_to_node.get_nodes();

				for (auto i = 1; i < nodes_on_path.size() - 1; i++) {
					const auto id = nodes_on_path[i];
					local_betweenness[id] += portion;
				}
			}
		}
	}

public:
	static double compute_average_betweenness_centrality(const DistributedGraph& graph) {
		const auto my_rank = MPIWrapper::get_my_rank();

		const auto number_local_nodes = graph.get_number_local_nodes();
		const auto total_number_nodes = NodeCounter::all_count_nodes(graph);

		const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(graph);
		const auto prefix_distribution = calculate_prefix_sum(node_distribution);

		auto status_counter = 0;
		const auto status_step = number_local_nodes / 101.0;

		std::vector<double> local_betweenness(total_number_nodes, 0.0);

		graph.lock_all_rma_windows();

		for (auto node_id = 0U; node_id < number_local_nodes; node_id++) {
			compute_betweenness_centrality(graph, node_id, total_number_nodes, prefix_distribution, local_betweenness);

			if (node_id % static_cast<int>(status_step) == 0 && status_counter < 100) {
				status_counter++;
				Status::report_status(node_id, total_number_nodes, "Betweenness Centrality");
			}
		}

		MPIWrapper::barrier();

		graph.unlock_all_rma_windows();

		Status::report_status(number_local_nodes, total_number_nodes, "Betweenness Centrality");

		const auto & result = MPIWrapper::reduce_componentwise(local_betweenness);

		const auto betweenness_centrality = std::reduce(result.begin(), result.end(), 0.0, std::plus<double>{});
		const auto average_betweenness_centrality = betweenness_centrality / total_number_nodes;

		return average_betweenness_centrality;
	}
};
