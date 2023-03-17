#include "AlgorithmTests.h"
#include "CentralityApprox.cpp" // solved linking problem (maybe needed because file name != class name (?))

void AlgorithmTests::test_algorithm_parallelization(std::filesystem::path input_directory) {
	const int my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	std::string test_result;
	std::chrono::time_point time = std::chrono::high_resolution_clock::now();

	// Test AreaConnectivity algorithm parallelization
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_parallel;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_sequential_helge;
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connect_sequential;
	try {
		MPIWrapper::barrier();
		time = std::chrono::high_resolution_clock::now();
		area_connect_parallel = AreaConnectivity::compute_area_connectivity_strength(dg);
		MPIWrapper::barrier();
		std::chrono::duration<double, std::milli> duration(std::chrono::high_resolution_clock::now() - time);

		area_connect_sequential_helge =
		    AreaConnectivity::area_connectivity_strength_sequential_helge(dg); // Runtime errors
		MPIWrapper::barrier();
		area_connect_sequential = AreaConnectivity::area_connectivity_strength_sequential(dg);
		MPIWrapper::barrier();
		compare_area_connec_map(*area_connect_parallel, *area_connect_sequential);
		compare_area_connec_map(*area_connect_parallel, *area_connect_sequential_helge);
		compare_area_connec_map(*area_connect_sequential_helge, *area_connect_sequential);

		test_result =
		    "AreaConnectivity test completed in " + std::to_string(duration.count()) + " milliseconds";
	} catch (std::string error_code) {
		test_result = "AreaConnectivity Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl << std::endl << std::endl;

	// Test Histogram algorithm parallelization
	std::unique_ptr<Histogram::HistogramData> histogram_count_bins;
	std::unique_ptr<Histogram::HistogramData> histogram_count_bins_sequential;
	std::uint64_t bin_count = 50;
	try {
		MPIWrapper::barrier();
		time = std::chrono::high_resolution_clock::now();
		histogram_count_bins = Histogram::compute_edge_length_histogram_const_bin_count(dg, bin_count);
		MPIWrapper::barrier();
		std::chrono::duration<double, std::milli> duration(std::chrono::high_resolution_clock::now() - time);

		histogram_count_bins_sequential =
		    Histogram::compute_edge_length_histogram_const_bin_count_sequential(dg, bin_count);
		MPIWrapper::barrier();
		compare_edge_length_histogram(*histogram_count_bins, *histogram_count_bins_sequential, 1e-8);

		test_result =
		    "Count Bins Histogram test completed in " + std::to_string(duration.count()) + " milliseconds";
	} catch (std::string error_code) {
		test_result = "Count Bins Histogram Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl << std::endl << std::endl;

	std::unique_ptr<Histogram::HistogramData> histogram_width_bins;
	std::unique_ptr<Histogram::HistogramData> histogram_width_bins_sequential;
	double bin_width = 1;
	try {
		MPIWrapper::barrier();
		time = std::chrono::high_resolution_clock::now();
		histogram_width_bins = Histogram::compute_edge_length_histogram_const_bin_width(dg, bin_width);
		MPIWrapper::barrier();
		std::chrono::duration<double, std::milli> duration(std::chrono::high_resolution_clock::now() - time);

		histogram_width_bins_sequential =
		    Histogram::compute_edge_length_histogram_const_bin_width_sequential(dg, bin_width);
		compare_edge_length_histogram(*histogram_width_bins, *histogram_width_bins_sequential, 1e-8);

		test_result =
		    "Width Bins Histogram test completed " + std::to_string(duration.count()) + " milliseconds";
	} catch (std::string error_code) {
		test_result = "Width Bins Histogram Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl << std::endl << std::endl;

	// Test Modularity algorithm parallelization
	double modularity_par, modularity_seq;
	try {
		MPIWrapper::barrier();
		time = std::chrono::high_resolution_clock::now();
		modularity_par = Modularity::compute_modularity(dg);
		MPIWrapper::barrier();
		std::chrono::duration<double, std::milli> duration(std::chrono::high_resolution_clock::now() - time);

		modularity_seq = Modularity::compute_modularity_sequential(dg);
		MPIWrapper::barrier();
		double absolute_error = std::abs(modularity_par - modularity_seq);
		double relative_error = absolute_error / (0.5 * (modularity_par + modularity_seq));
		if (relative_error > 1e-8) {
			std::stringstream error_code;
			error_code << "modularity_par:" << modularity_par << "   modularity_seq:" << modularity_seq
				   << "    absolute_error:" << absolute_error << "   relative_error:" << relative_error;
			throw error_code.str();
		}

		test_result = "Modularity test completed " + std::to_string(duration.count()) + " milliseconds";
	} catch (std::string error_code) {
		test_result = "Modularity Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl << std::endl << std::endl;

	// Test Network Motif algorithm parallelization
	std::array<long double, 14> motifs_par, motifs_seq;
	try {
		MPIWrapper::barrier();
		time = std::chrono::high_resolution_clock::now();
		motifs_par = NetworkMotifs::compute_network_TripleMotifs(dg);
		MPIWrapper::barrier();
		std::chrono::duration<double, std::milli> duration(std::chrono::high_resolution_clock::now() - time);

		motifs_seq = NetworkMotifs::compute_network_TripleMotifs_SingleProc(dg);
		MPIWrapper::barrier();

		if (my_rank == 0) {
			std::array<long double, 14> comp;
			for (int j = 0; j < comp.size(); j++) {
				comp[j] = motifs_par[j] - motifs_seq[j];
			}
			double absolute_perc_sum_error = std::accumulate(comp.begin() + 1, comp.end(), 0);
			double absolute_count_error = comp[0];
			double relative_error =
			    absolute_perc_sum_error + absolute_count_error / (0.5 * (motifs_par[0] + motifs_seq[0]));
			if (relative_error > 1e-8) {
				std::stringstream error_code;
				error_code << std::endl << "par:";
				for (long double m : motifs_par)
					error_code << m << "|";
				error_code << std::endl;
				error_code << "seq:";
				for (long double m : motifs_seq)
					error_code << m << "|";
				error_code << std::endl;
				throw error_code.str();
			}
		}

		test_result = "NetworkMotifs test completed " + std::to_string(duration.count()) + " milliseconds";
	} catch (std::string error_code) {
		test_result = "NetworkMotifs Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl << std::endl << std::endl;
}

void AlgorithmTests::test_centrality_approx(std::filesystem::path input_directory) {
	const int my_rank = MPIWrapper::get_my_rank();
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	std::string test_result;

	// Test BetweennessCentrality Approximation
	std::unique_ptr<BetweennessCentralityApproximation::BC_e> betweenness_centrality;
	int m = 20;
	double d = 0.25;
	int k = 10;

	unsigned int src_id = 0;
	unsigned int dest_id = 5;
	const auto total_number_nodes = NodeCounter::all_count_nodes(dg);
	const auto node_distribution = NodeDistributionCounter::all_count_node_distribution(dg);
	const auto prefix_distribution = calculate_prefix_sum(node_distribution);

	std::cout << "Start betweennessCentralityApprox..." << std::endl;
	try {
		MPIWrapper::barrier();

		if (my_rank == 0) {

			// Print local number of nodes of rank 0:
			const auto number_local_nodes = dg.get_number_local_nodes();
			std::cout << "rank0: number_local_nodes = " << number_local_nodes << std::endl;

			// Print all out_edges of rank 0:
			for (size_t i = 0; i < number_local_nodes; i++) {
				const auto& out_edges = dg.get_out_edges(0, i);
				for (OutEdge out_edge : out_edges) {
					std::cout << "out_edge[node" << i << "] = " << out_edge.target_rank << ","
						  << out_edge.target_id << " (" << out_edge.weight << ")" << std::endl;
				}
			}

			// Print all in_edges of rank 0:
			for (size_t i = 0; i < number_local_nodes; i++) {
				const auto& in_edges = dg.get_in_edges(0, i);
				for (InEdge in_edge : in_edges) {
					std::cout << "in_edge[node" << i << "] = " << in_edge.source_rank << ","
						  << in_edge.source_id << " (" << in_edge.weight << ")" << std::endl;
				}
			}

			std::vector<std::vector<NodePath>> ssp = BetweennessCentralityApproximation::compute_sssp(
			    dg, src_id, dest_id, total_number_nodes, prefix_distribution);
			// Print NodePath
			std::cout << "Print SSP: " << std::endl;
			for (size_t i = 0; i < ssp.size(); i++) {

				std::cout << i << ". spp: " << std::endl;
				for (size_t j = 0; j < ssp[i].size(); j++) {

					std::cout << "path = { ";
					for (size_t k = 0; k < ssp[i][j].get_nodes().size(); k++) {

						std::cout << ssp[i][j].get_nodes().at(k) << ", ";
					}
					std::cout << "}" << std::endl;
				}
			}
		}
		MPIWrapper::barrier();

		// betweenness_centrality =
		// BetweennessCentralityApproximation::compute_betweenness_centrality_approx(dg, m, d, k);
		// MPIWrapper::barrier();

		test_result = "BetweennessCentralityApprox test completed";

	} catch (std::string error_code) {
		test_result = "BetweennessCentralityApprox Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl;
}

void AlgorithmTests::check_graph_property(std::filesystem::path input_directory) {
	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	const int my_rank = MPIWrapper::get_my_rank();
	const int result_rank = 0;
	int number_ranks = MPIWrapper::get_number_ranks();
	std::uint64_t number_local_nodes = dg.get_number_local_nodes();
	std::vector<std::uint64_t> number_nodes_of_ranks(number_ranks);
	MPIWrapper::gather<uint64_t>(&number_local_nodes, number_nodes_of_ranks.data(), 1, MPI_UINT64_T, result_rank);
	MPIWrapper::barrier();

	// Check if total edge numbers equal
	uint64_t total_number_out_edges = OutEdgeCounter::count_out_edges(dg);
	uint64_t total_number_in_edges = InEdgeCounter::count_in_edges(dg);
	if (my_rank == result_rank) {
		std::cout << "Check if total edge numbers are equal:" << std::endl;
		std::cout << "\ttotal_number_out_edges = " << total_number_out_edges << std::endl;
		std::cout << "\ttotal_number_in_edges = " << total_number_in_edges << std::endl;
		if (total_number_out_edges != total_number_in_edges)
			std::cout << "\t(!) Total edge numbers do not equal" << std::endl;
	}
	MPIWrapper::barrier();
	if (my_rank == result_rank)
		std::cout << std::endl;

	// Check if there are nodes with self-referencing edges
	if (my_rank == result_rank)
		std::cout << "Check if there are nodes with self-referencing edges (list occurrences):" << std::endl;
	for (std::uint64_t node = 0; node < number_local_nodes; node++) {
		auto out_edges = dg.get_out_edges(my_rank, node);
		for (auto out_edge : out_edges) {
			if (out_edge.target_rank == my_rank && out_edge.target_id == node) {
				std::cout << "\t(!) Node with self referencing out_edge found: node(" << my_rank << ", "
					  << node << ")" << std::endl;
			}
		}
		auto in_edges = dg.get_in_edges(my_rank, node);
		for (auto in_edge : in_edges) {
			if (in_edge.source_rank == my_rank && in_edge.source_id == node) {
				std::cout << "\t(!) Node with self referencing in_edge found: node(" << my_rank << ", "
					  << node << ")" << std::endl;
			}
		}
	}
	MPIWrapper::barrier();
	if (my_rank == result_rank)
		std::cout << std::endl;

	// Check if there are edge duplicates
	if (my_rank == result_rank)
		std::cout << "Check if there are edge duplicates (list occurrences):" << std::endl;

	std::unordered_map<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uint64_t, std::uint64_t>>,
			   int, StdDoublePair_hash>
	    out_edge_count;
	std::unordered_map<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uint64_t, std::uint64_t>>,
			   int, StdDoublePair_hash>
	    in_edge_count;

	for (std::uint64_t node = 0; node < number_local_nodes; node++) {

		const std::vector<OutEdge>& out_edges = dg.get_out_edges(my_rank, node);
		const std::vector<InEdge>& in_edges = dg.get_in_edges(my_rank, node);

		for (const OutEdge& out_edge : out_edges) {
			std::pair<std::uint64_t, std::uint64_t> p1(my_rank, node);
			std::pair<std::uint64_t, std::uint64_t> p2(out_edge.target_rank, out_edge.target_id);
			std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uint64_t, std::uint64_t>>
			    key = {p1, p2};
			int& value = out_edge_count[key];
			if (value != 0) {
				std::cout << "\t(!) Out-edge duplicate found: (" << my_rank << ", " << node << ") "
					  << "---> (" << out_edge.target_rank << ", " << out_edge.target_id << ")"
					  << std::endl;
			}
			value++;
		}

		for (const InEdge& in_edge : in_edges) {
			std::pair<std::uint64_t, std::uint64_t> p1(my_rank, node);
			std::pair<std::uint64_t, std::uint64_t> p2(in_edge.source_rank, in_edge.source_id);
			std::pair<std::pair<std::uint64_t, std::uint64_t>, std::pair<std::uint64_t, std::uint64_t>>
			    key = {p1, p2};
			int& value = in_edge_count[key];
			if (value != 0) {
				std::cout << "\t(!) In-edge duplicate found: (" << my_rank << ", " << node << ") "
					  << "---> (" << in_edge.source_rank << ", " << in_edge.source_id << ")"
					  << std::endl;
			}
			value++;
		}
	}
	MPIWrapper::barrier();
	if (my_rank == result_rank)
		std::cout << std::endl;
}

void AlgorithmTests::compare_area_connec_map(const AreaConnectivity::AreaConnecMap& map_par,
					     const AreaConnectivity::AreaConnecMap& map_seq) {
	for (auto key_value = map_par.begin(); key_value != map_par.end(); key_value++) {
		auto other_key_value = map_seq.find(key_value->first);
		if (other_key_value != map_seq.end()) {
			if (other_key_value->second != key_value->second) {
				std::stringstream error_code;
				error_code << "key_value:" << key_value->first.first << " --> "
					   << key_value->first.second << "  map_par:" << key_value->second
					   << "  map_seq:" << other_key_value->second;
				throw error_code.str();
			}
		} else {
			std::stringstream error_code;
			error_code << "key_value:" << key_value->first.first << " --> " << key_value->first.second
				   << "  does not exist in map_seq";
			throw error_code.str();
		}
	}
	for (auto key_value = map_seq.begin(); key_value != map_seq.end(); key_value++) {
		auto other_key_value = map_par.find(key_value->first);
		if (other_key_value != map_par.end()) {
			if (other_key_value->second != key_value->second) {
				std::stringstream error_code;
				error_code << "key_value:" << key_value->first.first << " --> "
					   << key_value->first.second << "  map_seq:" << key_value->second
					   << "  map_par:" << other_key_value->second;
				throw error_code.str();
			}
		} else {
			std::stringstream error_code;
			error_code << "key_value:" << key_value->first.first << " --> " << key_value->first.second
				   << "  does not exist in map_par";
			throw error_code.str();
		}
	}

	if (map_par.size() != map_seq.size()) {
		std::stringstream error_code;
		error_code << "map_par:" << map_par.size() << " || "
			   << "map_seq:" << map_seq.size();
		throw error_code.str();
	}
}

void AlgorithmTests::compare_edge_length_histogram(const Histogram::HistogramData& histogram_par,
						   const Histogram::HistogramData& histogram_seq,
						   const double epsilon) {
	const auto my_rank = MPIWrapper::get_my_rank();
	if (my_rank != 0)
		return;

	if (histogram_par.size() != histogram_seq.size()) {
		std::stringstream error_code;
		error_code << "histogram_par.size():" << histogram_par.size() << "  histogram_par.size()"
			   << histogram_par.size();
		throw error_code.str();
	}

	std::uint64_t total_edges_par = 0;
	for (auto entry : histogram_par) {
		total_edges_par += entry.second;
	}

	std::uint64_t total_edges_seq = 0;
	for (auto entry : histogram_seq) {
		total_edges_seq += entry.second;
	}

	if (total_edges_par != total_edges_seq) {
		std::stringstream error_code;
		error_code << "total_edges_par:" << total_edges_par << "  total_edges_seq:" << total_edges_seq;
		throw error_code.str();
	}

	for (int bin = 0; bin < histogram_par.size(); bin++) {
		auto elem_par = histogram_par[bin];
		auto elem_seq = histogram_seq[bin];
		if (fabs(elem_par.first.first - elem_seq.first.first) > epsilon) {
			std::stringstream error_code;
			error_code << "Histograms have different bin boundings in bin:" << bin
				   << "  elem_par->first.first:" << elem_par.first.first
				   << "   elem_seq->first.first:" << elem_seq.first.first;
			throw error_code.str();
		}
		if (fabs(elem_par.first.second - elem_seq.first.second) > epsilon) {
			std::stringstream error_code;
			error_code << "Histograms have different bin boundings in bin:" << bin
				   << "  elem_par->first.second:" << elem_par.first.second
				   << "   elem_seq->first.second:" << elem_seq.first.second;
			throw error_code.str();
		}
		if (elem_par.second != elem_seq.second) {
			std::stringstream error_code;
			error_code << "Histograms have different bin values in bin:" << bin
				   << "  elem_par.second:" << elem_par.second
				   << "   elem_seq.second:" << elem_seq.second;
			throw error_code.str();
		}
	}
}
