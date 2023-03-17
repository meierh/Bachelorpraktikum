#include "NetworkMotifs.h"

std::array<long double, 14> NetworkMotifs::compute_network_triple_motifs(DistributedGraph& graph, unsigned int result_rank) {
	
	// Testing function parameters
	const int my_rank = MPIWrapper::get_my_rank();
	const std::uint64_t number_local_nodes = graph.get_number_local_nodes();
	const int number_ranks = MPIWrapper::get_number_ranks();

	if (result_rank >= number_ranks) {
		throw std::invalid_argument("Bad parameter - result_rank:" + result_rank);
	}
	graph.lock_all_rma_windows();

    std::chrono::time_point time = std::chrono::high_resolution_clock::now();
    std::vector<std::chrono::duration<double, std::milli>> times;
    std::vector<std::string> names;

	// Calculate the networkMotifs
	std::function<std::unique_ptr<std::vector<std::tuple<std::uint64_t, std::uint64_t, ThreeMotifStructure>>>(const DistributedGraph& dg,
														  std::uint64_t node_local_ind)>
	    collect_possible_network_motifs_one_node = [](const DistributedGraph& dg, std::uint64_t node_local_ind) {
		    
		    const int my_rank = MPIWrapper::get_my_rank();
		    const std::vector<OutEdge>& out_edges = dg.get_out_edges(my_rank, node_local_ind);
		    const std::vector<InEdge>& in_edges = dg.get_in_edges(my_rank, node_local_ind);

		    auto this_node_possible_motifs = std::make_unique<std::vector<std::tuple<std::uint64_t, std::uint64_t, ThreeMotifStructure>>>();
		    this_node_possible_motifs->reserve(out_edges.size() * in_edges.size());

		    auto& ref_this_node_possible_motifs = *this_node_possible_motifs;

		    // Compute an array of unique connected nodes with an bool pair wether the connecting edges are out,
		    // in or both
		    std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>>> adjacent_nodes_list;
		    adjacent_nodes_list.reserve(out_edges.size() + in_edges.size());

		    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, int, StdPair_hash> adjacent_nodes_to_index;
		    adjacent_nodes_to_index.reserve(out_edges.size());

		    for (const OutEdge& out_edge : out_edges) {
			    if (!(out_edge.target_rank == my_rank && out_edge.target_id == node_local_ind)) {

				    std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>> node_information(
					std::pair<std::uint64_t, std::uint64_t>(out_edge.target_rank, out_edge.target_id), std::pair<bool, bool>(true, false));

				    if (adjacent_nodes_to_index.find(node_information.first) == adjacent_nodes_to_index.end()) {
					    
					    adjacent_nodes_list.push_back(node_information);
					    adjacent_nodes_to_index[node_information.first] = adjacent_nodes_list.size() - 1;
				    }
			    }
		    }

		    for (const InEdge& in_edge : in_edges) {
			    if (!(in_edge.source_rank == my_rank && in_edge.source_id == node_local_ind)) {

				    std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>> node_information(
					std::pair<std::uint64_t, std::uint64_t>(in_edge.source_rank, in_edge.source_id), std::pair<bool, bool>(false, true));

				    auto entry = adjacent_nodes_to_index.find(node_information.first);

				    if (entry == adjacent_nodes_to_index.end()) {
					    adjacent_nodes_list.push_back(node_information);

				    } else {
					    assert(entry->second < adjacent_nodes_list.size());
					    std::pair<bool, bool>& edges_out_in = adjacent_nodes_list[entry->second].second;
					    assert(edges_out_in.first);
					    edges_out_in.second = true;
				    }
			    }
		    }

		    // Iterate over all node triples connected to node to compute possible motif types
		    for (int i = 0; i < adjacent_nodes_list.size(); i++) {

			    std::pair<std::uint64_t, std::uint64_t>& node_outer = adjacent_nodes_list[i].first;
			    std::pair<bool, bool> edges_out_in_to_node_outer = adjacent_nodes_list[i].second;

			    for (int j = 0; j < adjacent_nodes_list.size(); j++) {
				    if (i == j) {
					    continue;
				    }

				    std::pair<std::uint64_t, std::uint64_t>& node_inner = adjacent_nodes_list[j].first;
				    std::pair<bool, bool> edgesOutIn_to_node_Inner = adjacent_nodes_list[j].second;

				    ThreeMotifStructure motif_struc;

				    bool node_2_exists_outEdge = edges_out_in_to_node_outer.first;
				    bool node_2_exists_inEdge = edges_out_in_to_node_outer.second;

				    motif_struc.node_3_rank = node_inner.first;
				    motif_struc.node_3_local = node_inner.second;
				    bool node_3_exists_outEdge = edgesOutIn_to_node_Inner.first;
				    bool node_3_exists_inEdge = edgesOutIn_to_node_Inner.second;

				    std::uint8_t exists_edge_bit_array = 0;
				    exists_edge_bit_array |= node_2_exists_outEdge ? 1 : 0;
				    exists_edge_bit_array |= node_2_exists_inEdge ? 2 : 0;
				    exists_edge_bit_array |= node_3_exists_outEdge ? 4 : 0;
				    exists_edge_bit_array |= node_3_exists_inEdge ? 8 : 0;

				    switch (exists_edge_bit_array) {
				    case 5:
					    // three node motif 3 & 5 & 8 (0101)
					    motif_struc.set_motif_types({3, 5, 8});
					    break;
				    case 6:
					    // three node motif 10 (0110)
					    motif_struc.set_motif_types({10});
					    break;
				    case 7:
					    // three node motif 6 (0111)
					    motif_struc.set_motif_types({6});
					    break;
				    case 9:
					    // three node motif 2 & 7 (1001)
					    motif_struc.set_motif_types({2, 7});
					    break;
				    case 10:
					    // three node motif 1 & 11 (1010)
					    motif_struc.set_motif_types({1, 11});
					    break;
				    case 11:
					    // three node motif 4 (1011)
					    motif_struc.set_motif_types({4});
					    break;
				    case 13:
					    continue;
				    case 14:
					    continue;
				    case 15:
					    // three node motif 9 & 12 & 13 (1111)
					    motif_struc.set_motif_types({9, 12, 13});
					    break;
				    default:
					    assert(false);
				    }
				    std::tuple<std::uint64_t, std::uint64_t, ThreeMotifStructure> possible_motif = {node_outer.first, node_outer.second,
														    motif_struc};
				    ref_this_node_possible_motifs.push_back(possible_motif);
			    }
		    }
		    return std::move(this_node_possible_motifs);
	    };

	std::function<ThreeMotifStructure(const DistributedGraph& dg, std::uint64_t node_local_ind, ThreeMotifStructure para)>
	    evaluate_correct_network_motifs_one_node = [](const DistributedGraph& dg, std::uint64_t node_local_ind, ThreeMotifStructure possible_motif) {
		    
		    const int my_rank = MPIWrapper::get_my_rank();
		    const std::vector<OutEdge>& out_edges = dg.get_out_edges(my_rank, node_local_ind);
		    const std::vector<InEdge>& in_edges = dg.get_in_edges(my_rank, node_local_ind);

		    // Compute an map of unique connected nodes with an bool pair wether the connecting edges are
		    // out, in or both
		    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>, StdPair_hash> adjacent_nodes;

		    for (const OutEdge& out_edge : out_edges) {
			    std::pair<std::uint64_t, std::uint64_t> node_key(out_edge.target_rank, out_edge.target_id);
			    std::pair<bool, bool>& value = adjacent_nodes[node_key];
			    value.first = true;
		    }

		    for (const InEdge& in_edge : in_edges) {
			    std::pair<std::uint64_t, std::uint64_t> node_key(in_edge.source_rank, in_edge.source_id);
			    std::pair<bool, bool>& value = adjacent_nodes[node_key];
			    value.second = true;
		    }

		    // Being on node 2 of the possible_motif check which connection to node 3 exist and decide on
		    // the correct motif
		    std::pair<std::uint64_t, std::uint64_t> node_3_key(possible_motif.node_3_rank, possible_motif.node_3_local);
		    std::pair<bool, bool> edges_connected = {false, false};

		    if (adjacent_nodes.find(node_3_key) != adjacent_nodes.end()) {
			    edges_connected = adjacent_nodes[node_3_key];
		    }

		    bool exists_edge_node2_to_node3 = edges_connected.first;
		    bool exists_edge_node3_to_node2 = edges_connected.second;

		    if (exists_edge_node2_to_node3 && exists_edge_node3_to_node2) {
			    // edges between node 2 and 3 in both directions
			    // maintain motifs 8,10,11,13
			    possible_motif.unset_motif_types({1, 2, 3, 4, 5, 6, 7, 9, 12});

		    } else if (exists_edge_node2_to_node3 && !exists_edge_node3_to_node2) {
			    // only edge from node 2 to node 3
			    // maintain motifs 5,7
			    possible_motif.unset_motif_types({1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13});

		    } else if (!exists_edge_node2_to_node3 && exists_edge_node3_to_node2) {
			    // only edge from node 3 to node 2
			    // maintain motifs 12
			    possible_motif.unset_motif_types({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13});

		    } else {
			    // no edges between node 2 and 3
			    // maintain motifs 1,2,3,4,6,9
			    possible_motif.unset_motif_types({5, 7, 8, 10, 11, 12, 13});
		    }
		    assert(possible_motif.check_validity());
		    return possible_motif;
	    };

	std::unique_ptr<CommunicationPatterns::NodeToNodeQuestionStructure<ThreeMotifStructure, ThreeMotifStructure>> three_motif_results;

	three_motif_results = CommunicationPatterns::node_to_node_question<ThreeMotifStructure, ThreeMotifStructure>(
	    graph, MPIWrapper::MPI_threeMotifStructure, collect_possible_network_motifs_one_node, MPIWrapper::MPI_threeMotifStructure,
	    evaluate_correct_network_motifs_one_node);

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("ComputeMotifsCodes");
    time = std::chrono::high_resolution_clock::now();

	// Count the number of motifs locally
	std::array<std::uint64_t, 14> motif_type_count;
	motif_type_count.fill(0);

	for (std::uint64_t node_local_ind = 0; node_local_ind < number_local_nodes; node_local_ind++) {
		std::unique_ptr<std::vector<ThreeMotifStructure>> this_node_motifs_results;
		this_node_motifs_results = three_motif_results->getAnswersOfQuestionerNode(node_local_ind);

		for (int i = 0; i < this_node_motifs_results->size(); i++) {
			ThreeMotifStructure& one_motif = (*this_node_motifs_results)[i];

			for (int motif_type = 1; motif_type < 14; motif_type++) {
				
				if (one_motif.is_motif_type_set(motif_type)) {
					motif_type_count[motif_type]++;
				}
			}
		}
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CollectMotifNumber");
    time = std::chrono::high_resolution_clock::now();

	// Collect the number of motifs globally
	std::array<std::uint64_t, 14> motif_type_count_total;
	MPIWrapper::reduce<std::uint64_t>(motif_type_count.data(), motif_type_count_total.data(), 14, MPI_UINT64_T, MPI_SUM, result_rank);

	// Reduce the motifs that were counted multiple times due to their invariant nature
	if (my_rank == result_rank) {

		// Order invariant motifs were counted two times each
		assert(motif_type_count_total[1] % 2 == 0);
		motif_type_count_total[1] /= 2;
		assert(motif_type_count_total[3] % 2 == 0);
		motif_type_count_total[3] /= 2;
		assert(motif_type_count_total[8] % 2 == 0);
		motif_type_count_total[8] /= 2;
		assert(motif_type_count_total[9] % 2 == 0);
		motif_type_count_total[9] /= 2;
		assert(motif_type_count_total[11] % 2 == 0);
		motif_type_count_total[11] /= 2;

		// Rotational invariant motifs were counted three times each
		assert(motif_type_count_total[7] % 3 == 0);
		motif_type_count_total[7] /= 3;

		// Order and Rotational invariant motifs were counted six times each
		assert(motif_type_count_total[13] % 6 == 0);
		motif_type_count_total[13] /= 6;
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("CollMotifNumGlobal");
    time = std::chrono::high_resolution_clock::now();

	// Compute the resulting array of motif numbers
	std::array<long double, 14> motif_fraction;
	if (my_rank == result_rank) {

		std::uint64_t total_number_of_motifs = std::accumulate(motif_type_count_total.begin(), motif_type_count_total.end(), 0);
		motif_fraction[0] = total_number_of_motifs;

		for (int motif_type = 1; motif_type < 14; motif_type++) {
			motif_fraction[motif_type] = motif_type_count_total[motif_type];
			// motif_fraction[motif_type] = static_cast<long double>(motif_type_count_total[motif_type]) /
			// static_cast<long double>(total_number_of_motifs);
		}
	}

    times.push_back(std::chrono::high_resolution_clock::now() - time);
    names.push_back("ComputDistribution");
    time = std::chrono::high_resolution_clock::now();

	graph.unlock_all_rma_windows();

    std::vector<double> time_double;
    std::for_each(times.cbegin(), times.cend(), [&](auto time) { time_double.push_back(time.count()); });

    std::vector<double> global_avg_times_double(6);
    MPIWrapper::reduce<double>(time_double.data(), global_avg_times_double.data(), 6, MPI_DOUBLE, MPI_SUM, 0);
    std::for_each(global_avg_times_double.begin(), global_avg_times_double.end(), [=](double& time) { time /= number_ranks; });

    std::vector<double> global_max_times_double(6);
    MPIWrapper::reduce<double>(time_double.data(), global_max_times_double.data(), 6, MPI_DOUBLE, MPI_MAX, 0);

    if (my_rank == 0) {
	    std::cout.precision(5);
	    std::cout << "compute_networkMotifs" << std::endl;
	    for (int i = 0; i < names.size(); i++) {
		    std::cout << names[i] << ":\tavg:" << global_avg_times_double[i] << "\tmax:" << global_max_times_double[i] << "   milliseconds"
			      << std::endl;
	    }
	    double total_avg = std::accumulate(global_avg_times_double.begin(), global_avg_times_double.end(), 0);
	    double total_max = std::accumulate(global_max_times_double.begin(), global_max_times_double.end(), 0);
	    std::cout << "Total             "
		      << ":\tavg:" << total_avg << "\tmax:" << total_max << "   milliseconds" << std::endl;
	    std::cout << "----------------------------------" << std::endl;
	    fflush(stdout);
    }

	return motif_fraction;
}

std::array<long double, 14> NetworkMotifs::compute_network_triple_motifs_sequential(DistributedGraph& graph, unsigned int result_rank) {
	
	const int my_rank = MPIWrapper::get_my_rank();
	int number_ranks = MPIWrapper::get_number_ranks();
	std::uint64_t number_local_nodes = graph.get_number_local_nodes();
	auto total_number_nodes = NodeCounter::all_count_nodes(graph);

	// Main rank gathers other ranks number of nodes
	std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
	MPIWrapper::gather<uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);

	// Prepare structure
	std::vector<std::uint64_t> motif_type_count(14, 0);

	if (my_rank == result_rank) {

		for (int current_rank = 0; current_rank < number_ranks; current_rank++) {
			for (std::uint64_t current_node = 0; current_node < number_nodes_of_ranks[current_rank]; current_node++) {

				// Gather information of adjacent nodes
				const std::vector<OutEdge>& out_edges = graph.get_out_edges(current_rank, current_node);
				const std::vector<InEdge>& in_edges = graph.get_in_edges(current_rank, current_node);

				std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>, StdPair_hash> adjacent_nodes;

				for (const OutEdge& out_edge : out_edges) {
					std::pair<std::uint64_t, std::uint64_t> node_key(out_edge.target_rank, out_edge.target_id);
					std::pair<bool, bool>& value = adjacent_nodes[node_key];
					assert(!value.first);
					value.first = true;
				}

				for (const InEdge& in_edge : in_edges) {
					std::pair<std::uint64_t, std::uint64_t> node_key(in_edge.source_rank, in_edge.source_id);
					std::pair<bool, bool>& value = adjacent_nodes[node_key];
					assert(!value.second);
					value.second = true;
				}

				for (auto iter_outer = adjacent_nodes.begin(); iter_outer != adjacent_nodes.end(); iter_outer++) {
					// Prrevent overlapping of outer and inner
					// -> consider only each tripple-node-set (ignore permutations)
					std::pair<std::uint64_t, std::uint64_t> node_outer_key = iter_outer->first;
					auto iter_inner = iter_outer;
					iter_inner++;

					for (iter_inner; iter_inner != adjacent_nodes.end(); iter_inner++) {

						std::pair<std::uint64_t, std::uint64_t> node_inner_key = iter_inner->first;
						if (node_inner_key != node_outer_key) {
							// Exclude nodes with self-referencing edges
							if ((current_rank == node_outer_key.first && current_node == node_outer_key.second) ||
							    (current_rank == node_inner_key.first && current_node == node_inner_key.second) ||
							    (node_outer_key.first == node_inner_key.first && node_outer_key.second == node_inner_key.second)) {
								continue;
							}

							ThreeMotifStructure motif_struc;
							motif_struc.node_3_rank = node_inner_key.first;
							motif_struc.node_3_local = node_inner_key.second;

							bool exists_edge_node1_to_node2 = iter_outer->second.first;
							bool exists_edge_node2_to_node1 = iter_outer->second.second;
							bool exists_edge_node1_to_node3 = iter_inner->second.first;
							bool exists_edge_node3_to_node1 = iter_inner->second.second;

							std::uint16_t exists_edge_bit_array = 0;
							exists_edge_bit_array |= exists_edge_node1_to_node2 ? 1 : 0;
							exists_edge_bit_array |= exists_edge_node2_to_node1 ? 2 : 0;
							exists_edge_bit_array |= exists_edge_node1_to_node3 ? 4 : 0;
							exists_edge_bit_array |= exists_edge_node3_to_node1 ? 8 : 0;

							std::uint16_t exists_edge_bit_array_updated =
							    update_edge_bit_array(graph, exists_edge_bit_array, node_outer_key.first, node_outer_key.second,
										 motif_struc.node_3_rank, motif_struc.node_3_local);

							switch (exists_edge_bit_array) {
							case 10:
								// three node motif 1 & 11 (0101)
								if (exists_edge_bit_array_updated == 10)
									motif_struc.set_motif_types({1});
								else if (exists_edge_bit_array_updated == 58)
									motif_struc.set_motif_types({11});
								break;

							case 5:
								// three node motif 3 & 8 (1010)
								if (exists_edge_bit_array_updated == 5)
									motif_struc.set_motif_types({3});
								else if (exists_edge_bit_array_updated == 53)
									motif_struc.set_motif_types({8});
								break;

							case 6:
							case 9:
								// three node motif 2 & 5 & 7 & 10 (0110, 1001)
								if (exists_edge_bit_array_updated == 6 || exists_edge_bit_array_updated == 9)
									motif_struc.set_motif_types({2});
								else if (exists_edge_bit_array_updated == 22 || exists_edge_bit_array_updated == 41)
									motif_struc.set_motif_types({5});
								else if (exists_edge_bit_array_updated == 38 || exists_edge_bit_array_updated == 25)
									motif_struc.set_motif_types({7});
								else if (exists_edge_bit_array_updated == 54 || exists_edge_bit_array_updated == 57)
									motif_struc.set_motif_types({10});
								else
									std::cout << "error: case 6/9 -> bitArray = " << exists_edge_bit_array_updated
										  << std::endl;
								break;

							case 14:
							case 11:
								// three node motif 4 (0111, 1101)
								if (exists_edge_bit_array_updated == 14 || exists_edge_bit_array_updated == 11)
									motif_struc.set_motif_types({4});
								break;

							case 7:
							case 13:
								// three node motif 6 (1110, 1011)
								if (exists_edge_bit_array_updated == 7 || exists_edge_bit_array_updated == 13)
									motif_struc.set_motif_types({6});
								break;

							case 15:
								// three node motif 9 & 12 & 13 (1111)
								if (exists_edge_bit_array_updated == 15)
									motif_struc.set_motif_types({9});
								else if (exists_edge_bit_array_updated == 31 || exists_edge_bit_array_updated == 47)
									motif_struc.set_motif_types({12});
								else if (exists_edge_bit_array_updated == 63)
									motif_struc.set_motif_types({13});
								else
									std::cout << "error: case 15 -> bitArray = " << exists_edge_bit_array_updated
										  << std::endl;
								break;

							default:
								break;
							}

							// Count every motif
							for (int motif_type = 1; motif_type < 14; motif_type++) {
								if (motif_struc.is_motif_type_set(motif_type)) {
									motif_type_count[motif_type]++;
								}
							}
						}
					}
				}
			}
		}
		// Rotational invariant motifs where counted three times each
		if (motif_type_count[7] % 3 != 0)
			std::cout << "error: motif_type_count[7]%3 != 0 ==> " << motif_type_count[7] << std::endl;

		motif_type_count[7] /= 3;

		if (motif_type_count[13] % 3 != 0)
			std::cout << "error: motif_type_count[13]%3 != 0 ==> " << motif_type_count[13] << std::endl;

		motif_type_count[13] /= 3;
	}

	std::array<long double, 14> motif_fraction;
	if (my_rank == result_rank) {

		std::uint64_t total_number_of_motifs = std::accumulate(motif_type_count.begin(), motif_type_count.end(), 0);
		motif_fraction[0] = total_number_of_motifs;

		for (int motif_type = 1; motif_type < 14; motif_type++) {
			motif_fraction[motif_type] = motif_type_count[motif_type];
			// motif_fraction[motif_type] = static_cast<long double>(motif_type_count[motif_type]) /
			// static_cast<long double>(total_number_of_motifs);
		}
	}

	MPIWrapper::barrier();
	return motif_fraction;
}

std::uint16_t NetworkMotifs::update_edge_bit_array(const DistributedGraph& graph, std::uint16_t exists_edge_bit_array, unsigned int node_2_rank,
						  std::uint64_t node_2_local, unsigned int node_3_rank, std::uint64_t node_3_local) {
	const std::vector<OutEdge>& out_edges = graph.get_out_edges(node_2_rank, node_2_local);
	const std::vector<InEdge>& in_edges = graph.get_in_edges(node_2_rank, node_2_local);

	std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::pair<bool, bool>, StdPair_hash> adjacent_nodes;

	for (const OutEdge& out_edge : out_edges) {
		std::pair<std::uint64_t, std::uint64_t> node_key(out_edge.target_rank, out_edge.target_id);
		std::pair<bool, bool>& value = adjacent_nodes[node_key];
		assert(!value.first);
		value.first = true;
	}

	for (const InEdge& in_edge : in_edges) {
		std::pair<std::uint64_t, std::uint64_t> node_key(in_edge.source_rank, in_edge.source_id);
		std::pair<bool, bool>& value = adjacent_nodes[node_key];
		assert(!value.second);
		value.second = true;
	}

	std::pair<std::uint64_t, std::uint64_t> node_3_key(node_3_rank, node_3_local);
	std::pair<bool, bool>& value = adjacent_nodes[node_3_key];

	bool exists_edge_node2_to_node3 = value.first;
	bool exists_edge_node3_to_node2 = value.second;

	exists_edge_bit_array |= exists_edge_node2_to_node3 ? 16 : 0;
	exists_edge_bit_array |= exists_edge_node3_to_node2 ? 32 : 0;

	return exists_edge_bit_array;
}

void NetworkMotifs::ThreeMotifStructure::self_test() {
	for (int motif = 1; motif < 14; motif++) {
		set_motif_types({motif});
		unset_motif_types({motif});
		assert(motif_type_bit_array == 0);
	}

	for (int motif = 1; motif < 14; motif++) {
		set_motif_types({motif});
	}

	for (int motif = 1; motif < 14; motif++) {
		unset_motif_types({motif});
	}

	assert(motif_type_bit_array == 0);
}

void NetworkMotifs::ThreeMotifStructure::set_motif_types(std::vector<int> motif_types) {
	for (int motif_type : motif_types) {
		assert(motif_type >= 1 && motif_type < 14);
		motif_type_bit_array |= (1 << motif_type);
	}
	// std::cout<<"  Set "<<motif_types[0]<<" ";
	// print_out();
}

void NetworkMotifs::ThreeMotifStructure::unset_motif_types(std::vector<int> motif_types) {
	std::uint64_t cp_motif_type_bit_array = motif_type_bit_array;
	for (int motif_type : motif_types) {
		assert(motif_type >= 1 && motif_type < 14);
		motif_type_bit_array &= ~(1 << motif_type);
	}
	/*
	std::cout<<"Unset "<<motif_types[0]<<" ";
	print_out();

	if(!check_validity())
	{
	    std::cout<<"prev-------------";
	    for(int i=1;i<14;i++)
		if(cp_motif_type_bit_array & (1<<i))
		    std::cout<<"1"<<" ";
		else
		    std::cout<<"0"<<" ";
	    std::cout<<"--------------------"<<std::endl;
	    for(int a:motif_types)
		std::cout<<a<<",";
	    std::cout<<std::endl;
	}
	*/
}

void NetworkMotifs::ThreeMotifStructure::unset_all_but_motif_types(std::vector<int> motif_types) {
	std::unordered_set<int> maintained_motif_types;
	maintained_motif_types.insert(motif_types.begin(), motif_types.end());
	std::vector<int> motif_types_to_unset;

	for (int motif_type = 1; motif_type < 14; motif_type++) {
		if (maintained_motif_types.find(motif_type) == maintained_motif_types.end()) {
			motif_types_to_unset.push_back(motif_type);
		}
	}
	unset_motif_types(motif_types_to_unset);
}

bool NetworkMotifs::ThreeMotifStructure::is_motif_type_set(int motif_type) {
	assert(motif_type >= 1 && motif_type < 14);
	return motif_type_bit_array & (1 << motif_type);
}

void NetworkMotifs::ThreeMotifStructure::print_out_complete() {
	std::cout << "(" << node_3_rank << "," << node_3_local << ")";
	print_out();
}

void NetworkMotifs::ThreeMotifStructure::print_out() {
	std::cout << "---";
	for (int i = 1; i < 14; i++)
		if (is_motif_type_set(i))
			std::cout << "1"
				  << " ";
		else
			std::cout << "0"
				  << " ";
	std::cout << "--------------------" << std::endl;
}

bool NetworkMotifs::ThreeMotifStructure::check_validity() {
	bool res = motif_type_bit_array && !(motif_type_bit_array & (motif_type_bit_array - 1));
	res |= motif_type_bit_array == 0;
	if (!res) {
		print_out();
	}
	return res;
}
