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

	const auto& rank_prefix = std::string("rank_") + std::to_string(my_rank);
	const auto& positions_file = rank_prefix + "_positions.txt";
	const auto& in_edges_file = rank_prefix + "_in_edges.txt";
	const auto& out_edges_file = rank_prefix + "_out_edges.txt";

	const auto& positions = path / positions_file;
	const auto& in_edges = path / in_edges_file;
	const auto& out_edges = path / out_edges_file;

	if (!std::filesystem::exists(positions)) {
		std::cout << "The positions for rank " << my_rank << " do not exist:\n" << positions << std::endl;
		throw 1;
	}

	if (!std::filesystem::exists(in_edges)) {
		std::cout << "The in edges for rank " << my_rank << " do not exist:\n" << in_edges << std::endl;
		throw 2;
	}

	if (!std::filesystem::exists(out_edges)) {
		std::cout << "The out edges for rank " << my_rank << " do not exist:\n" << out_edges << std::endl;
		throw 3;
	}

	load_nodes(positions);
	load_in_edges(in_edges);
	load_out_edges(out_edges);

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
	std::vector<int> area_names_ind{};
	std::vector<int> signal_types_ind{};
	
	std::unordered_map<std::string,int> area_names_set;
	std::unordered_map<std::string,int> signal_types_set;

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
		
		const auto area_name_pos = area_names_set.find(area_name);
		if(area_name_pos != area_names_set.end())
		//Area name already encountered
		{
			area_names_ind.emplace_back(area_name_pos->second);
		}
		else
		//Area name new
		{
			area_names_set.insert(std::pair<std::string,int>(area_name,area_names.size()));
			area_names_ind.emplace_back(area_names.size());
			area_names.push_back(area_name);
		}
		
		const auto signal_type_pos = signal_types_set.find(area_name);
		if(signal_type_pos != signal_types_set.end())
		//Signal name already encountered
		{
			signal_types_ind.emplace_back(signal_type_pos->second);
		}
		else
		//signal type new
		{
			signal_types_set.insert(std::pair<std::string,int>(signal_type,signal_types.size()));
			signal_types_ind.emplace_back(signal_types.size());
			signal_types.push_back(signal_type);
		}
	}

	nodes_window = MPIWrapper::create_rma_window<Vec3d>(positions.size());
	local_number_nodes = positions.size();
	
	area_names_ind_window = MPIWrapper::create_rma_window<std::uint64_t>(area_names.size());
	signal_types_ind_window = MPIWrapper::create_rma_window<std::uint64_t>(signal_types.size());

	const auto my_rank = MPIWrapper::get_my_rank();

	MPIWrapper::lock_window_exclusive(my_rank, nodes_window.window);
	std::memcpy(nodes_window.my_base_pointer, positions.data(), sizeof(Vec3d) * positions.size());
	MPIWrapper::unlock_window(my_rank, nodes_window.window);
	
	MPIWrapper::lock_window_exclusive(my_rank, area_names_ind_window.window);
	std::memcpy(area_names_ind_window.my_base_pointer, area_names_ind.data(), sizeof(int) * area_names_ind.size());
	MPIWrapper::unlock_window(my_rank, area_names_ind_window.window);
	
	MPIWrapper::lock_window_exclusive(my_rank, signal_types_ind_window.window);
	std::memcpy(signal_types_ind_window.my_base_pointer, signal_types_ind.data(), sizeof(int) * signal_types_ind.size());
	MPIWrapper::unlock_window(my_rank, signal_types_ind_window.window);
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
