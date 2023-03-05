#include "AreaConnectivity.h"

std::unique_ptr<AreaConnectivity::AreaConnecMap>
AreaConnectivity::compute_area_connectivity_strength(DistributedGraph& graph, const unsigned int result_rank) {
	graph.lock_all_rma_windows();
	MPIWrapper::barrier();

	// Test function parameters
	const int number_of_ranks = MPIWrapper::get_number_ranks();
	if (result_rank >= number_of_ranks) {
		throw std::invalid_argument("Bad parameter - result_rank:" + result_rank);
	}

	// Build local area connection map
	const int my_rank = MPIWrapper::get_my_rank();
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
	std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t, std::uint64_t, AreaConnectivityInfo>>>(
	    const DistributedGraph&, std::uint64_t)>
	    transfer_connection_sources = [&](const DistributedGraph& dg, std::uint64_t node_local_ind) {
		    const std::vector<OutEdge>& out_edges = dg.get_out_edges(my_rank, node_local_ind);
		    std::int64_t node_areaID = dg.get_node_area_localID(my_rank, node_local_ind);
		    auto connection_sources =
			std::make_unique<std::vector<std::tuple<std::uint64_t, std::uint64_t, AreaConnectivityInfo>>>(
			    out_edges.size());
		    for (int i = 0; i < out_edges.size(); i++) {
			    AreaConnectivityInfo one_connection = {my_rank, node_areaID, out_edges[i].target_rank, -1,
								   out_edges[i].weight};
			    (*connection_sources)[i] =
				std::tie(out_edges[i].target_rank, out_edges[i].target_id, one_connection);
		    }
		    return std::move(connection_sources);
	    };
	std::function<AreaConnectivityInfo(const DistributedGraph&, std::uint64_t, AreaConnectivityInfo)>
	    fill_connection_target_areaID =
		[&](const DistributedGraph& dg, std::uint64_t node_local_ind, AreaConnectivityInfo connection) {
			connection.target_area_localID = dg.get_node_area_localID(my_rank, node_local_ind);
			return connection;
		};
	std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<AreaConnectivityInfo, AreaConnectivityInfo>>
	    area_connections = CommunicationPatterns::node_to_node_question<AreaConnectivityInfo, AreaConnectivityInfo>(
		graph, MPIWrapper::MPI_AreaConnectivityInfo, transfer_connection_sources,
		MPIWrapper::MPI_AreaConnectivityInfo, fill_connection_target_areaID);

	AreaIDConnecMap my_rank_connecID_map;
	for (std::uint64_t node_ID = 0; node_ID < number_local_nodes; node_ID++) {
		std::unique_ptr<std::vector<AreaConnectivityInfo>> node_connections =
		    area_connections->getAnswersOfQuestionerNode(node_ID);
		for (AreaConnectivityInfo& one_connection : *node_connections) {
			AreaLocalID source_nameID(one_connection.source_rank, one_connection.source_area_localID);
			AreaLocalID target_nameID(one_connection.target_rank, one_connection.target_area_localID);
			my_rank_connecID_map[{source_nameID, target_nameID}] += one_connection.weight;
		}
	}

	// Gather local area connection maps to result rank
	using connecID_map_data = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
	std::function<std::unique_ptr<std::vector<std::pair<connecID_map_data, int>>>(const DistributedGraph&)>
	    extract_connecID_map = [&](const DistributedGraph& dg) {
		    auto connecID_list = std::make_unique<std::vector<std::pair<connecID_map_data, int>>>();
		    for (auto map_entry = my_rank_connecID_map.cbegin(); map_entry != my_rank_connecID_map.cend();
			 map_entry++) {
			    auto connecID_data_entry = std::tie(
				map_entry->first.first.first, map_entry->first.first.second,
				map_entry->first.second.first, map_entry->first.second.second, map_entry->second);
			    std::pair<connecID_map_data, int> composed_Entry(connecID_data_entry,
									     std::tuple_size<connecID_map_data>());
			    connecID_list->push_back(composed_Entry);
		    }
		    return connecID_list;
	    };
	std::unique_ptr<std::vector<std::vector<connecID_map_data>>> ranks_to_connecID_data =
	    CommunicationPatterns::gather_Data_to_one_Rank<connecID_map_data, std::int64_t>(
		graph, extract_connecID_map,
		[](connecID_map_data dat) {
			auto [s_r, s_n, t_r, t_n, w] = dat;
			return std::vector<std::int64_t>({s_r, s_n, t_r, t_n, w});
		},
		[](std::vector<std::int64_t>& data_vec) {
			assert(data_vec.size() == std::tuple_size<connecID_map_data>());
			return std::tie(data_vec[0], data_vec[1], data_vec[2], data_vec[3], data_vec[4]);
		},
		MPI_INT64_T, result_rank);

	// Gather name lists from all ranks to result rank
	std::function<std::unique_ptr<std::vector<std::pair<std::string, int>>>(const DistributedGraph&)> get_names =
	    [](const DistributedGraph& dg) {
		    const std::vector<std::string>& area_names = dg.get_local_area_names();
		    auto data = std::make_unique<std::vector<std::pair<std::string, int>>>(area_names.size());
		    std::transform(area_names.cbegin(), area_names.cend(), data->begin(),
				   [](std::string name) { return std::pair<std::string, int>(name, name.size()); });
		    return std::move(data);
	    };
	std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks =
	    CommunicationPatterns::gather_Data_to_one_Rank<std::string, char>(
		graph, get_names,
		[](std::string area_name) { return std::vector<char>(area_name.cbegin(), area_name.cend()); },
		[](std::vector<char> area_name_vec) { return std::string(area_name_vec.data(), area_name_vec.size()); },
		MPI_CHAR, result_rank);

	// Combine connecID_maps and transfer IDs to area names
	auto global_connec_name_map = std::make_unique<AreaConnecMap>();
	for (const std::vector<connecID_map_data>& connec_data_of_rank : *ranks_to_connecID_data) {
		for (const connecID_map_data& connec_data : connec_data_of_rank) {
			assert(std::get<0>(connec_data) < area_names_list_of_ranks->size());
			assert(std::get<1>(connec_data) < (*area_names_list_of_ranks)[std::get<0>(connec_data)].size());
			std::string source_area =
			    (*area_names_list_of_ranks)[std::get<0>(connec_data)][std::get<1>(connec_data)];
			assert(std::get<2>(connec_data) < area_names_list_of_ranks->size());
			assert(std::get<3>(connec_data) < (*area_names_list_of_ranks)[std::get<2>(connec_data)].size());
			std::string target_area =
			    (*area_names_list_of_ranks)[std::get<2>(connec_data)][std::get<3>(connec_data)];
			(*global_connec_name_map)[{source_area, target_area}] += std::get<4>(connec_data);
		}
	}

	MPIWrapper::barrier();
	graph.unlock_all_rma_windows();

	return std::move(global_connec_name_map);
}

std::unique_ptr<AreaConnectivity::AreaConnecMap>
AreaConnectivity::area_connectivity_strength_sequential_helge(const DistributedGraph& graph, unsigned int result_rank) {
	const int my_rank = MPIWrapper::get_my_rank();
	const int number_of_ranks = MPIWrapper::get_number_ranks();
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();

	std::function<std::unique_ptr<std::vector<std::pair<std::uint64_t, int>>>(const DistributedGraph& dg)>
	    get_number_nodes = [&](const DistributedGraph& dg) {
		    auto number_nodes = std::make_unique<std::vector<std::pair<std::uint64_t, int>>>();
		    number_nodes->push_back(std::pair<std::uint64_t, int>(number_local_nodes, 1));
		    return number_nodes;
	    };
	std::unique_ptr<std::vector<std::vector<std::uint64_t>>> ranks_to_number_local_nodes =
	    CommunicationPatterns::gather_Data_to_one_Rank<std::uint64_t, std::uint64_t>(
		graph, get_number_nodes, [](std::uint64_t dat) { return std::vector<std::uint64_t>({dat}); },
		[](std::vector<std::uint64_t>& data_vec) { return data_vec[0]; }, MPI_UINT64_T, result_rank);

	std::function<std::unique_ptr<std::vector<std::pair<std::string, int>>>(const DistributedGraph&)> get_names =
	    [](const DistributedGraph& dg) {
		    const std::vector<std::string>& area_names = dg.get_local_area_names();
		    auto data = std::make_unique<std::vector<std::pair<std::string, int>>>(area_names.size());
		    std::transform(area_names.cbegin(), area_names.cend(), data->begin(),
				   [](std::string name) { return std::pair<std::string, int>(name, name.size()); });
		    return std::move(data);
	    };
	std::unique_ptr<std::vector<std::vector<std::string>>> area_names_list_of_ranks =
	    CommunicationPatterns::gather_Data_to_one_Rank<std::string, char>(
		graph, get_names,
		[](std::string area_name) { return std::vector<char>(area_name.cbegin(), area_name.cend()); },
		[](std::vector<char> area_name_vec) { return std::string(area_name_vec.data(), area_name_vec.size()); },
		MPI_CHAR, result_rank);

	auto global_connec_name_map = std::make_unique<AreaConnecMap>();
	if (my_rank == result_rank) {
		for (int rank = 0; rank < number_of_ranks; rank++) {
			for (std::uint64_t node_local_ind = 0; node_local_ind < (*ranks_to_number_local_nodes)[rank][0];
			     node_local_ind++) {
				std::int64_t source_node_areaID = graph.get_node_area_localID(rank, node_local_ind);
				const std::vector<OutEdge>& out_edges = graph.get_out_edges(rank, node_local_ind);
				for (const OutEdge& out_edge : out_edges) {
					std::int64_t target_node_areaID =
					    graph.get_node_area_localID(out_edge.target_rank, out_edge.target_id);
					std::string source_area = (*area_names_list_of_ranks)[rank][source_node_areaID];
					std::string target_area =
					    (*area_names_list_of_ranks)[out_edge.target_rank][target_node_areaID];
					(*global_connec_name_map)[{source_area, target_area}] += out_edge.weight;
				}
			}
		}
	}

	return std::move(global_connec_name_map);
}

std::unique_ptr<AreaConnectivity::AreaConnecMap>
AreaConnectivity::area_connectivity_strength_sequential(const DistributedGraph& graph, unsigned int result_rank) {
	// Define all relevant local variables
	const int& my_rank = MPIWrapper::get_my_rank();
	const int& number_ranks = MPIWrapper::get_number_ranks();
	const int& number_local_nodes = graph.get_number_local_nodes();
	std::vector<uint64_t> node_numbers = MPIWrapper::all_gather(graph.get_number_local_nodes());
	const std::vector<std::string> area_names = graph.get_local_area_names();
	const int& number_area_names = area_names.size();
	MPIWrapper::barrier();

	// ========== GET AREA NAMES TO SINGLE COMPUTATION RANK ==========

	// Create local transmit_area_names_string and area_names_char_len vectors
	std::vector<int> area_names_char_len;
	std::vector<char> transmit_area_names_string;
	for (const std::string& name : area_names) {
		for (int i = 0; i < name.size(); i++) {
			transmit_area_names_string.push_back(name[i]);
		}
		area_names_char_len.push_back(name.size());
	}
	MPIWrapper::barrier();

	// Gather rank_to_number_area_names vector as helper for char_len_displ (and global_area_names_char_len)
	assert(area_names_char_len.size() == number_area_names); // debug
	int nbr_area_names = area_names.size();
	std::vector<int> rank_to_number_area_names;
	if (my_rank == result_rank) {
		rank_to_number_area_names.resize(number_ranks);
	}
	MPIWrapper::gather<int>(&nbr_area_names, rank_to_number_area_names.data(), 1, MPI_INT, result_rank);
	MPIWrapper::barrier();

	// Crate char_len_displ as helper for global_area_names_char_len
	// and prepare global_area_names_char_len with correct size
	std::vector<int> char_len_displ;
	std::vector<int> global_area_names_char_len;
	if (my_rank == result_rank) {
		char_len_displ.resize(number_ranks);
		int displacement = 0;

		for (int r = 0; r < number_ranks; r++) {
			char_len_displ[r] = displacement;
			displacement += rank_to_number_area_names[r];
		}
		global_area_names_char_len.resize(displacement);
	}
	MPIWrapper::barrier();

	// Gather global_area_names_char_len (with Help of char_len_displ and rank_to_number_area_names)
	MPIWrapper::gatherv<int>(area_names_char_len.data(), nbr_area_names, global_area_names_char_len.data(),
				 rank_to_number_area_names.data(), char_len_displ.data(), MPI_INT, result_rank);
	MPIWrapper::barrier();

	// Finally create rank_to_area_names_char_len as helper for rank_to_area_names
	std::vector<std::vector<int>> rank_to_area_names_char_len;
	if (my_rank == result_rank) {
		rank_to_area_names_char_len.resize(number_ranks);
		for (int r = 0; r < number_ranks - 1; r++) {
			for (int l = char_len_displ[r]; l < char_len_displ[r + 1]; l++) {
				rank_to_area_names_char_len[r].push_back(global_area_names_char_len[l]);
			}
		}
		for (int l = char_len_displ[number_ranks - 1]; l < global_area_names_char_len.size(); l++) {
			rank_to_area_names_char_len[number_ranks - 1].push_back(global_area_names_char_len[l]);
		}
	}
	MPIWrapper::barrier();

	// Gather rank_to_string_len as helper for char_displ
	int nbr_string_chars = transmit_area_names_string.size();
	std::vector<int> rank_to_string_len;
	if (my_rank == result_rank) {
		rank_to_string_len.resize(number_ranks);
	}
	MPIWrapper::gather<int>(&nbr_string_chars, rank_to_string_len.data(), 1, MPI_INT, result_rank);
	MPIWrapper::barrier();

	// Create char_displ as helper for rank_to_area_names
	std::vector<int> char_displ;
	if (my_rank == result_rank) {
		char_displ.resize(number_ranks);
		int displacement = 0;
		for (int r = 0; r < number_ranks; r++) {
			char_displ[r] = displacement;
			displacement += rank_to_string_len[r];
		}
	}
	MPIWrapper::barrier();

	// Prepare and gather global_area_names_string as helper for rank_to_area_names
	std::vector<char> global_area_names_string;
	if (my_rank == result_rank) {
		int sum = std::accumulate(rank_to_string_len.begin(), rank_to_string_len.end(), 0);
		global_area_names_string.resize(sum);
	}
	MPIWrapper::gatherv<char>(transmit_area_names_string.data(), nbr_string_chars, global_area_names_string.data(),
				  rank_to_string_len.data(), char_displ.data(), MPI_CHAR, result_rank);
	MPIWrapper::barrier();

	// Finally create rank_to_area_names
	std::vector<std::vector<std::string>> rank_to_area_names;
	if (my_rank == result_rank) {
		rank_to_area_names.resize(number_ranks);
		int displacement = 0;
		for (int r = 0; r < rank_to_area_names_char_len.size(); r++) {
			for (int l = 0; l < rank_to_area_names_char_len[r].size(); l++) {
				std::string name(&global_area_names_string[displacement],
						 rank_to_area_names_char_len[r][l]);
				rank_to_area_names[r].push_back(name);
				displacement += rank_to_area_names_char_len[r][l];
			}
		}
	}
	MPIWrapper::barrier();

	// ========== LET SINGLE RANK COMPUTE AREA CONNECTIVITY ==========

	auto result = std::make_unique<AreaConnecMap>();

	// Computation is performed by a single process:
	if (my_rank == result_rank) {
		// Iterate over each rank...
		for (int r = 0; r < number_ranks; r++) {
			// ...and over each node from that rank
			for (int n = 0; n < node_numbers[r]; n++) {
				// Consider each outgoing edge and find out source and target areas
				const std::vector<OutEdge>& out_edges = graph.get_out_edges(r, n);
				for (const OutEdge& out_edge : out_edges) {
					std::uint64_t source_area_localID = graph.get_node_area_localID(r, n);
					std::uint64_t target_area_localID =
					    graph.get_node_area_localID(out_edge.target_rank, out_edge.target_id);
					std::string source_area_str = rank_to_area_names[r][source_area_localID];
					std::string target_area_str =
					    rank_to_area_names[out_edge.target_rank][target_area_localID];

					// Store all weights of the area pairs that realize a connection of two
					// different areas in the corresponding "area to area hash class" of the result
					// map
					(*result)[{source_area_str, target_area_str}] += out_edge.weight;
				}
			}
		}
	}
	return std::move(result);
}
