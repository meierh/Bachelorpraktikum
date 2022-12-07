#include "DistributedGraph.h"

#include "MPIWrapper.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>	
#include <vector>

DistributedGraph::DistributedGraph(const std::filesystem::path& path) {
	const auto my_rank = MPIWrapper::get_my_rank();

	int number_of_digits_ranks = std::to_string(MPIWrapper::get_number_ranks() - 1).length();
	int number_of_digits_my_rank = std::to_string(my_rank).length();

  	// zeros = number_of_ranks - rank (Stellen)
  	int number_of_zeros = number_of_digits_ranks - number_of_digits_my_rank;
  	std::string zeros = "";
	// append number of zeros needed
  	while (number_of_zeros != 0) {
    	zeros = zeros + "0";
		number_of_zeros--;
  	}

	const std::string& rank_prefix = std::string("rank_") + zeros + std::to_string(my_rank);

	const auto& positions_file = rank_prefix + "_positions.txt";
	const auto& in_network_file = rank_prefix + "_in_network.txt";
	const auto& out_network_file = rank_prefix + "_out_network.txt";

	const std::string& network_string = "network";
	const std::string& positions_string = "positions";

	const auto& positions = path / positions_string / positions_file;
	const auto& in_network = path / network_string / in_network_file;
	const auto& out_network = path / network_string / out_network_file;

	if (!std::filesystem::exists(positions)) {
		std::cout << "The positions for rank " << my_rank << " do not exist:\n" << positions << std::endl;
		throw 1;
	}

	if (!std::filesystem::exists(in_network)) {
		std::cout << "The in edges for rank " << my_rank << " do not exist:\n" << in_network << std::endl;
		throw 2;
	}

	if (!std::filesystem::exists(out_network)) {
		std::cout << "The out edges for rank " << my_rank << " do not exist:\n" << out_network << std::endl;
		throw 3;
	}

	load_nodes(positions);
	load_in_edges(in_network);
	load_out_edges(out_network);

	in_edge_cache.init();
	out_edge_cache.init();
}

void DistributedGraph::lock_all_rma_windows() const {
	const auto error_1 = MPI_Win_lock_all(0, nodes_window.window);
	const auto error_2 = MPI_Win_lock_all(0, in_edges_window.window);
	const auto error_3 = MPI_Win_lock_all(0, prefix_in_edges_window.window);
	const auto error_4 = MPI_Win_lock_all(0, number_in_edges_window.window);
	const auto error_5 = MPI_Win_lock_all(0, out_edges_window.window);
	const auto error_6 = MPI_Win_lock_all(0, prefix_out_edges_window.window);
	const auto error_7 = MPI_Win_lock_all(0, number_out_edges_window.window);
}

void DistributedGraph::unlock_all_rma_windows() const {
	const auto error_1 = MPI_Win_unlock_all(nodes_window.window);
	const auto error_2 = MPI_Win_unlock_all(in_edges_window.window);
	const auto error_3 = MPI_Win_unlock_all(prefix_in_edges_window.window);
	const auto error_4 = MPI_Win_unlock_all(number_in_edges_window.window);
	const auto error_5 = MPI_Win_unlock_all(out_edges_window.window);
	const auto error_6 = MPI_Win_unlock_all(prefix_out_edges_window.window);
	const auto error_7 = MPI_Win_unlock_all(number_out_edges_window.window);
}

void DistributedGraph::load_nodes(const std::filesystem::path& path) {
	std::ifstream file(path);

	const bool file_is_good = file.good();
	const bool file_is_not_good = file.fail() || file.eof();

	std::string line{};

	std::vector<Vec3d> positions{};

	while (std::getline(file, line)) {
		if (line.empty() || '#' == line[0]) {
			continue;
		}

		std::istringstream iss{ line };

		std::uint64_t id{ 0 };

		double pos_x{ 0.0 };
		double pos_y{ 0.0 };
		double pos_z{ 0.0 };

		std::string area_name{};
		std::string signal_type{};

		const auto success = (iss >> id) && (iss >> pos_x) && (iss >> pos_y) && (iss >> pos_z) && (iss >> area_name) && (iss >> signal_type);

		if (!success) {
			std::cout << "Error processing line: " << line << '\n';
			continue;
		}

		const auto expected_id = positions.size() + 1;

		if (expected_id != id) {
			std::cout << "Expected to load node with id " << expected_id << " but loaded id " << id << '\n';
			continue;
		}

		if (id >= std::numeric_limits<unsigned int>::max()) {
			std::cout << "The loaded id: " << id << " is larger than the maximum value of unsigned int.\n";
			continue;
		}

		positions.emplace_back(pos_x, pos_y, pos_z);
	}

	nodes_window = MPIWrapper::create_rma_window<Vec3d>(positions.size());
	local_number_nodes = positions.size();

	const auto my_rank = MPIWrapper::get_my_rank();

	MPIWrapper::lock_window_exclusive(my_rank, nodes_window.window);
	std::memcpy(nodes_window.my_base_pointer, positions.data(), sizeof(Vec3d) * positions.size());
	MPIWrapper::unlock_window(my_rank, nodes_window.window);
}

void DistributedGraph::load_in_edges(const std::filesystem::path& path) {
	std::ifstream file(path);

	const bool file_is_good = file.good();
	const bool file_is_not_good = file.fail() || file.eof();

	const auto my_rank = MPIWrapper::get_my_rank();

	std::string line{};
	std::map<unsigned int, std::vector<InEdge>> all_in_edges{};

	auto loaded_in_edges = 0;

	while (std::getline(file, line)) {
		if (line.empty() || '#' == line[0]) {
			continue;
		}

		std::istringstream iss{ line };

		int target_rank{};
		unsigned int target_id{};

		int source_rank{};
		unsigned int  source_id{};

		int weight{};

		const auto success = (iss >> target_rank) && (iss >> target_id) && (iss >> source_rank) && (iss >> source_id) && (iss >> weight);

		if (!success) {
			std::cout << "Error processing line: " << line << '\n';
			continue;
		}

		if (target_rank != my_rank) {
			std::cout << "Loading in-edges for rank " << my_rank << " but found an in-edge that is directed to rank " << target_rank << '\n';
			continue;
		}

		// IDs are 1-based
		target_id--;
		source_id--;

		if (target_id >= local_number_nodes) {
			std::cout << "Loaded an in-edge with target id " << target_id << " but I only have " << local_number_nodes << " nodes.\n";
			continue;
		}

		all_in_edges[target_id].emplace_back(source_rank, source_id, weight);
		loaded_in_edges++;
	}

	in_edges_window = MPIWrapper::create_rma_window<InEdge>(loaded_in_edges);
	local_number_in_edges = loaded_in_edges;

	std::vector<std::uint64_t> number_in_edges(local_number_nodes);
	std::vector<std::uint64_t> prefix_number_in_edges(local_number_nodes, 0);

	auto current_filling = 0;

	MPIWrapper::lock_window_exclusive(my_rank, in_edges_window.window);
	for (auto target_id = 0; target_id < local_number_nodes; target_id++) {
		const auto& in_edges = all_in_edges[target_id];

		std::memcpy(in_edges_window.my_base_pointer + current_filling, in_edges.data(), in_edges.size() * sizeof(InEdge));
		current_filling += in_edges.size();
		number_in_edges[target_id] = in_edges.size();

		if (target_id > 0) {
			prefix_number_in_edges[target_id] = prefix_number_in_edges[target_id - 1] + all_in_edges[target_id - 1].size();
		}
	}
	MPIWrapper::unlock_window(my_rank, in_edges_window.window);

	number_in_edges_window = MPIWrapper::create_rma_window<std::uint64_t>(local_number_nodes);

	MPIWrapper::lock_window_exclusive(my_rank, number_in_edges_window.window);
	std::memcpy(number_in_edges_window.my_base_pointer, number_in_edges.data(), sizeof(std::uint64_t) * local_number_nodes);
	MPIWrapper::unlock_window(my_rank, number_in_edges_window.window);

	prefix_in_edges_window = MPIWrapper::create_rma_window<std::uint64_t>(local_number_nodes);

	MPIWrapper::lock_window_exclusive(my_rank, prefix_in_edges_window.window);
	std::memcpy(prefix_in_edges_window.my_base_pointer, prefix_number_in_edges.data(), sizeof(std::uint64_t) * local_number_nodes);
	MPIWrapper::unlock_window(my_rank, prefix_in_edges_window.window);
}

void DistributedGraph::load_out_edges(const std::filesystem::path& path) {
	std::ifstream file(path);

	const bool file_is_good = file.good();
	const bool file_is_not_good = file.fail() || file.eof();

	const auto my_rank = MPIWrapper::get_my_rank();

	std::string line{};
	std::map<unsigned int, std::vector<InEdge>> all_out_edges{};

	auto loaded_out_edges = 0;

	while (std::getline(file, line)) {
		if (line.empty() || '#' == line[0]) {
			continue;
		}

		std::istringstream iss{ line };

		int target_rank{};
		unsigned int target_id{};

		int source_rank{};
		unsigned int  source_id{};

		int weight{};

		const auto success = (iss >> target_rank) && (iss >> target_id) && (iss >> source_rank) && (iss >> source_id) && (iss >> weight);

		if (!success) {
			std::cout << "Error processing line: " << line << '\n';
			continue;
		}

		if (source_rank != my_rank) {
			std::cout << "Loading out-edges for rank " << my_rank << " but found an out-edge that is directed to rank " << source_rank << '\n';
			continue;
		}

		// IDs are 1-based
		target_id--;
		source_id--;

		if (source_id >= local_number_nodes) {
			std::cout << "Loaded an out-edge with source id " << source_id << " but I only have " << local_number_nodes << " nodes.\n";
			continue;
		}

		all_out_edges[source_id].emplace_back(target_rank, target_id, weight);
		loaded_out_edges++;
	}

	out_edges_window = MPIWrapper::create_rma_window<OutEdge>(loaded_out_edges);
	local_number_out_edges = loaded_out_edges;

	std::vector<std::uint64_t> number_out_edges(local_number_nodes);
	std::vector<std::uint64_t> prefix_number_out_edges(local_number_nodes, 0);

	auto current_filling = 0;

	MPIWrapper::lock_window_exclusive(my_rank, out_edges_window.window);
	for (auto source_id = 0; source_id < local_number_nodes; source_id++) {
		const auto& out_edges = all_out_edges[source_id];

		std::memcpy(out_edges_window.my_base_pointer + current_filling, out_edges.data(), out_edges.size() * sizeof(OutEdge));
		current_filling += out_edges.size();

		number_out_edges[source_id] = out_edges.size();

		if (source_id > 0) {
			prefix_number_out_edges[source_id] = prefix_number_out_edges[source_id - 1] + +all_out_edges[source_id - 1].size();
		}
	}
	MPIWrapper::unlock_window(my_rank, out_edges_window.window);

	number_out_edges_window = MPIWrapper::create_rma_window<std::uint64_t>(local_number_nodes);

	MPIWrapper::lock_window_exclusive(my_rank, number_out_edges_window.window);
	std::memcpy(number_out_edges_window.my_base_pointer, number_out_edges.data(), sizeof(std::uint64_t) * local_number_nodes);
	MPIWrapper::unlock_window(my_rank, number_out_edges_window.window);

	prefix_out_edges_window = MPIWrapper::create_rma_window<std::uint64_t>(local_number_nodes);

	MPIWrapper::lock_window_exclusive(my_rank, prefix_out_edges_window.window);
	std::memcpy(prefix_out_edges_window.my_base_pointer, prefix_number_out_edges.data(), sizeof(std::uint64_t) * local_number_nodes);
	MPIWrapper::unlock_window(my_rank, prefix_out_edges_window.window);
}
