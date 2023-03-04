#include "DistributedGraph.h"
#include "MPIWrapper.h"

#include "AllPairsShortestPath.h"
#include "Centrality.h"
#include "CentralityApprox.h"
#include "Clustering.h"
#include "DegreeCounter.h"
#include "EdgeCounter.h"
#include "EdgeLength.h"
#include "NodeCounter.h"

#include "CommunicationPatterns.h"
#include "AlgorithmTests.h"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>

#include <csignal>
#include <iostream>
#include <sstream>
#include <string>

void calculate_metrics(std::filesystem::path input_directory) {
	const auto my_rank = MPIWrapper::get_my_rank();

	DistributedGraph dg(input_directory);

	MPIWrapper::barrier();

	if (0 == my_rank) {
		std::cout << "All " << MPIWrapper::get_number_ranks() << " ranks finished loading their local data!" << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto number_total_nodes = NodeCounter::count_nodes(dg);

	const auto number_total_in_edges = InEdgeCounter::count_in_edges(dg);
	const auto number_total_out_edges = OutEdgeCounter::count_out_edges(dg);

	const auto [min_in, max_in] = InDegreeCounter::count_in_degrees(dg);
	const auto [min_out, max_out] = OutDegreeCounter::count_out_degrees(dg);

	const auto average_edge_length = EdgeLength::compute_edge_length(dg);

	const auto [average_shortest_path, global_efficiency, number_unreachables] = AllPairsShortestPath::compute_apsp(dg);

	const auto average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg);

	const auto average_cluster_coefficient = Clustering::compute_average_clustering_coefficient(dg);

	if (my_rank == 0) {
		std::cout << "The total number of nodes in the graph is: " << number_total_nodes << '\n';
		std::cout << "The total number of in edges in the graph is: " << number_total_in_edges << '\n';
		std::cout << "The total number of out edges in the graph is: " << number_total_out_edges << '\n';
		std::cout << "The minimum in-degree is " << min_in << " and the maximum in-degree is " << max_in << '\n';
		std::cout << "The minimum out-degree is " << min_out << " and the maximum out-degree is " << max_out << '\n';
		std::cout << "The average edge length is " << average_edge_length << '\n';
		std::cout << "The average shortest path is " << average_shortest_path << ", however, " << number_unreachables << " pairs had no path.\n";
		std::cout << "The global efficiency is " << global_efficiency << ", however, " << number_unreachables << " pairs had no path.\n";
		std::cout << "The average clustering coefficient is: " << average_cluster_coefficient << '\n';
		std::cout << "The average betweenness centrality is: " << average_betweenness_centrality << '\n';
		fflush(stdout);
	}
}

void test_networkMotifs(std::filesystem::path input_directory) {
	const auto my_rank = MPIWrapper::get_my_rank();

	DistributedGraph dg(input_directory);
	MPIWrapper::barrier();
	
	// Test NetworkMotifs algorithm parallelization
	std::string test_result;
	std::array<long double,14> motifs_par, motifs_seq;
	try {
		motifs_par = NetworkMotifs::compute_network_TripleMotifs(dg);
		MPIWrapper::barrier();
		motifs_seq = NetworkMotifs::compute_network_TripleMotifs_SingleProc(dg);
		MPIWrapper::barrier();
		
		if(my_rank==0)
		{
			std::cout<<"par:";
			for(long double m : motifs_par)
				std::cout<<m<<"|";
			std::cout<<std::endl;

			std::cout<<"seq:";
			for(long double m : motifs_seq)
				std::cout<<m<<"|";
			std::cout<<std::endl;
		}

		test_result = "NetworkMotifs test completed";
	} catch (std::string error_code) {
		test_result = "NetworkMotifs Error :" + error_code;
	}
	if (my_rank == 0)
		std::cout << test_result << std::endl;
}
	
	
int main(int argument_count, char* arguments[]) {
	CLI::App app{ "" };

	std::string input_directory{};
	auto* opt_input_directory = app.add_option("--input", input_directory, "The directory that contains the input files.")->required();
	opt_input_directory->check(CLI::ExistingDirectory);

	CLI11_PARSE(app, argument_count, arguments);

	MPIWrapper::init(argument_count, arguments);

	//test_algorithm_parallelization(input_directory);
	test_networkMotifs(input_directory);

	MPIWrapper::finalize();

	return 0;
}
