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

//Struct to store all the CLI arguments bundled up
//	so we dont need to give them sperately as arguments
struct CLIArguments {
	std::filesystem::path input_directory;
	bool enable_all_pairs_shortest_path;
	bool enable_area_connectivity;
	bool enable_assortativity;
	bool enable_betweenness_centrality;
	bool enable_betweenness_centrality_approx;
	bool enable_clustering_coefficient;
	bool enable_histogram_width;
	bool enable_histogram_count;
	bool enable_modularity;
	bool enable_network_motifs;
};

void calculate_metrics(const CLIArguments& args) {
	const auto my_rank = MPIWrapper::get_my_rank();

	DistributedGraph dg(args.input_directory);

	MPIWrapper::barrier();

	if (0 == my_rank) {
		std::cout << "All " << MPIWrapper::get_number_ranks() << " ranks finished loading their local data!" << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto number_total_nodes = NodeCounter::count_nodes(dg);
	if (my_rank == 0) {
		std::cout << "The total number of nodes in the graph is: " << number_total_nodes << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto number_total_in_edges = InEdgeCounter::count_in_edges(dg);
	if (my_rank == 0) {
		std::cout << "The total number of in edges in the graph is: " << number_total_in_edges << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto number_total_out_edges = OutEdgeCounter::count_out_edges(dg);
	if (my_rank == 0) {
		std::cout << "The total number of out edges in the graph is: " << number_total_out_edges << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto [min_in, max_in] = InDegreeCounter::count_in_degrees(dg);
	if (my_rank == 0) {
		std::cout << "The minimum in-degree is " << min_in << " and the maximum in-degree is " << max_in << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto [min_out, max_out] = OutDegreeCounter::count_out_degrees(dg);
	if (my_rank == 0) {
		std::cout << "The minimum out-degree is " << min_out << " and the maximum out-degree is " << max_out << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	const auto average_edge_length = EdgeLength::compute_edge_length(dg);
	if (my_rank == 0) {
		std::cout << "The average edge length is " << average_edge_length << '\n';
		fflush(stdout);
	}

	MPIWrapper::barrier();

	double average_shortest_path, global_efficiency = 0.0;
	unsigned long number_unreachables = 0l;
	if(args.enable_all_pairs_shortest_path) {
		std::tie(average_shortest_path, global_efficiency, number_unreachables) = AllPairsShortestPath::compute_apsp(dg);

		if (my_rank == 0) {
			std::cout << "The average shortest path is " << average_shortest_path << ", however, " << number_unreachables << " pairs had no path.\n";
			std::cout << "The global efficiency is " << global_efficiency << ", however, " << number_unreachables << " pairs had no path.\n";
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	double average_betweenness_centrality = 0.0;
	if(args.enable_betweenness_centrality) {
		average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg);

		if (my_rank == 0) {
			std::cout << "The average betweenness centrality is: " << average_betweenness_centrality << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	double average_cluster_coefficient = 0.0;
	if(args.enable_clustering_coefficient) {
		average_cluster_coefficient = Clustering::compute_average_clustering_coefficient(dg);

		if (my_rank == 0) {
			std::cout << "The average clustering coefficient is: " << average_cluster_coefficient << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connectivity;
	if(args.enable_area_connectivity) {
		area_connectivity = AreaConnectivity::compute_area_connectivity_strength(dg);

		if (my_rank == 0) {
			std::cout << "The area connectivity connections are:\n";
			int nr = 0;
			for (auto& [key, value]: (*area_connectivity)) {
				std::cout << "Connection " << nr 
					  << ": weight = " << value << " (" << key.first << " --> " << key.second << ")" << std::endl;
				nr++;
			}
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	std::tuple<double, double, double, double> assortativity_coefficient;
	if(args.enable_assortativity) {
		assortativity_coefficient = Assortativity::compute_assortativity_coefficient(dg);

		if (my_rank == 0) {
			std::cout << "The assortativitiy coefficients are: \n"
				  << "r_in_in = " << std::get<0>(assortativity_coefficient) << "\n"
				  << "r_in_out = " << std::get<1>(assortativity_coefficient) << "\n"
				  << "r_out_in = " << std::get<2>(assortativity_coefficient) << "\n"
				  << "r_out_out = " << std::get<3>(assortativity_coefficient) << "\n"
			<< std::endl;

		}
		fflush(stdout);
		MPIWrapper::barrier();
	}

	double modularity = 0.0;
	if(args.enable_modularity) {
		modularity = Modularity::compute_modularity(dg);

		if (my_rank == 0) {
			std::cout << "The modularity is : " << modularity << std::endl;
		}
		fflush(stdout);
		MPIWrapper::barrier();
	}

	std::array<long double, 14> motifs;
	if(args.enable_network_motifs) {
		motifs = NetworkMotifs::compute_network_triple_motifs(dg);
		
		if (my_rank == 0) {
			std::cout << "The Motif fractions are:\n";
			for (size_t i = 1; i < motifs.size(); i++) {
				std::cout << "motif " << i << " = " << motifs.at(i) << std::endl;
			}
			std::cout << "The total count of motif occurrences is: " << motifs.at(0) << std::endl;
		}
		fflush(stdout);
		MPIWrapper::barrier();
	}

}
	
	
int main(int argument_count, char* arguments[]) {
	CLI::App app{ "" };
	CLIArguments args{};

	auto* opt_input_directory = app.add_option("--input", args.input_directory, "The directory that contains the input files.")->required();
	opt_input_directory->check(CLI::ExistingDirectory);

	std::vector<std::string> selected_algorithms{};
	app.add_flag("--func", selected_algorithms,
					"Select what functions to run as a comma seperated list"
					" [avg-apsp, area-connec, assortativity, avg-centrality, apx-centrality, \n avg-cluster, histogr-wid, histogr-cnt, modularity, tri-motifs]")
					->delimiter(',')->allow_extra_args(true);
	CLI11_PARSE(app, argument_count, arguments);

	for (const auto& arg: selected_algorithms) {
		if(arg == "avg-apsp") {
			args.enable_all_pairs_shortest_path = true;
		}else if(arg == "area-connec") {
			args.enable_area_connectivity = true;
		}else if(arg == "assortativity") {
			args.enable_assortativity = true;
		}else if(arg == "avg-centrality") {
			args.enable_betweenness_centrality = true;
	  	}else if(arg == "apx-centrality") {
			args.enable_betweenness_centrality_approx = true;
		}else if(arg == "avg-cluster") {
			args.enable_clustering_coefficient = true;
		}else if(arg == "histogr-wid") {
			args.enable_histogram_width = true;
		}else if(arg == "histogr-cnt") {
			args.enable_histogram_count = true;
		}else if(arg == "modularity") {
			args.enable_modularity = true;
		}else if(arg == "tri-motifs") {
			args.enable_network_motifs = true;
		}else {
			std::cout << "The argument \"" << arg << "\" is not a valid algorithm" << std::endl;
			//terminate the programm and return a failure exitcode
			return EXIT_FAILURE;
		}
	}

	MPIWrapper::init(argument_count, arguments);

	calculate_metrics(args);
	
	/* call AlgorithmTests: */
	//AlgorithmTests::check_graph_property(args.input_directory);
	AlgorithmTests::test_algorithm_parallelization(args.input_directory);

	MPIWrapper::finalize();

	return 0;
}
