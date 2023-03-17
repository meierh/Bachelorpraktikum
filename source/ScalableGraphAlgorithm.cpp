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
	bool enable_node_count;
	bool enable_edge_count;
	bool enable_degree_extr;
	bool enable_average_edge_length;
	bool enable_all_pairs_shortest_path;
	bool enable_clustering_coefficient;
	bool enable_betweenness_centrality_average;
	bool enable_betweenness_centrality_approx;
	bool enable_area_connectivity;
	bool enable_histogram_width;
	bool enable_histogram_count;
	bool enable_modularity;
	bool enable_network_motifs;
	bool enable_assortativity;
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

	if(args.enable_node_count) {
		const auto number_total_nodes = NodeCounter::count_nodes(dg);
		
		if (my_rank == 0) {
			std::cout << "The total number of nodes in the graph is: " << number_total_nodes << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_edge_count) {
		const auto number_total_in_edges = InEdgeCounter::count_in_edges(dg);
		const auto number_total_out_edges = OutEdgeCounter::count_out_edges(dg);
		
		if (my_rank == 0) {
			std::cout << "The total number of in edges in the graph is: " << number_total_in_edges << '\n';
			std::cout << "The total number of out edges in the graph is: " << number_total_out_edges << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_degree_extr) {
		const auto [min_in, max_in] = InDegreeCounter::count_in_degrees(dg);
		const auto [min_out, max_out] = OutDegreeCounter::count_out_degrees(dg);
		
		if (my_rank == 0) {
			std::cout << "The minimum in-degree is: " << min_in << " and the maximum in-degree is " << max_in << '\n';
			std::cout << "The minimum out-degree is: " << min_out << " and the maximum out-degree is " << max_out << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_average_edge_length){
		const auto average_edge_length = EdgeLength::compute_edge_length(dg);
		
		if (my_rank == 0) {
			std::cout << "The average edge length is: " << average_edge_length << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_all_pairs_shortest_path) {
		auto[average_shortest_path, global_efficiency, number_unreachables] = AllPairsShortestPath::compute_apsp(dg);
		
		if (my_rank == 0) {
			std::cout << "The average shortest path is: " << average_shortest_path << ", however, " << number_unreachables << " pairs had no path.\n";
			std::cout << "The global efficiency is: " << global_efficiency << ", however, " << number_unreachables << " pairs had no path.\n";
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_clustering_coefficient) {
		double average_cluster_coefficient = Clustering::compute_average_clustering_coefficient(dg);
		
		if (my_rank == 0) {
			std::cout << "The average clustering coefficient is: " << average_cluster_coefficient << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_betweenness_centrality_average) {
		double average_betweenness_centrality = BetweennessCentrality::compute_average_betweenness_centrality(dg);

		if (my_rank == 0) {
			std::cout << "The average betweenness centrality is: " << average_betweenness_centrality << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_betweenness_centrality_approx) {
		//int m; int k;
		//double d;
		//auto approx_betweenness_centrality = BetweennessCentralityApproximation::compute_betweenness_centrality_approx(dg, m, d, k);

		if (my_rank == 0) {
			std::cout << "The average betweenness centrality is: " << "Not available..." << '\n';
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	
	std::unique_ptr<AreaConnectivity::AreaConnecMap> area_connectivity;
	if(args.enable_area_connectivity) {
		area_connectivity = AreaConnectivity::compute_area_connectivity_strength(dg);

		if (my_rank == 0) {
			std::cout << "The area connectivity is given by:\n";
			int nr = 0;
			for (auto& [key, value]: (*area_connectivity)) {
				std::cout << "Connection " << nr 
					  << ": weight = " << value << " (" << key.first << " --> " << key.second << ")\n";
				nr++;
			}
			fflush(stdout);
		}
		MPIWrapper::barrier();
		
	}

	if(args.enable_histogram_width) {
		double width = 1;
		std::unique_ptr<Histogram::HistogramData> histogram_wid = Histogram::compute_edge_length_histogram_const_bin_width(dg, width);
		
		if (my_rank == 0) {
			std::cout << "The edge length histogram for the bin width " << width << " is given by:\n";
			for (int i = 0; i < histogram_wid->size(); i++) {
				std::cout << i << ". " << "bin: " << (*histogram_wid)[i].first.first << "-" << (*histogram_wid)[i].first.second << ": " << (*histogram_wid)[i].second << "\n";
			}
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_histogram_count) {
		uint64_t count = 50;
		std::unique_ptr<Histogram::HistogramData> histogram_cnt = Histogram::compute_edge_length_histogram_const_bin_count(dg, count);
		
		if (my_rank == 0) {
			std::cout << "The edge length histogram for the bin count " << count << " is given by:\n";
			for (int i = 0; i < histogram_cnt->size(); i++) {
				std::cout << i << ". " << "bin: " << (*histogram_cnt)[i].first.first << "-" << (*histogram_cnt)[i].first.second << ": " << (*histogram_cnt)[i].second << "\n";
			}
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_modularity) {
		double modularity = Modularity::compute_modularity(dg);

		if (my_rank == 0) {
			std::cout << "The modularity is : " << modularity << std::endl;
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_network_motifs) {
		std::array<long double, 14> motifs = NetworkMotifs::compute_network_triple_motifs(dg);
		
		if (my_rank == 0) {
			std::cout << "The normalized network motif counts are:\n";
			for (size_t i = 1; i < motifs.size(); i++) {
				std::cout << "motif " << i << " = " << motifs.at(i) << std::endl;
			}
			std::cout << "The total count of tripple-motif occurrences is: " << motifs.at(0) << std::endl;
			fflush(stdout);
		}
		MPIWrapper::barrier();
	}

	if(args.enable_assortativity) {
		std::tuple<double, double, double, double> assortativity_coefficient = Assortativity::compute_assortativity_coefficient(dg);

		if (my_rank == 0) {
			std::cout << "The assortativitiy coefficients are: \n"
				  << "r_in_in = " << std::get<0>(assortativity_coefficient) << "\n"
				  << "r_in_out = " << std::get<1>(assortativity_coefficient) << "\n"
				  << "r_out_in = " << std::get<2>(assortativity_coefficient) << "\n"
				  << "r_out_out = " << std::get<3>(assortativity_coefficient) << "\n";
			fflush(stdout);
		}
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
					"Select what functions to run as a comma seperated list:\n"
					"node-cnt,\nedge-cnt,\ndegree-ext,\navg-edge-len,\navg-apsp,\navg-cluster,\navg-centrality,\n"
					"apx-centrality,\narea-connec,\nhistogr-wid,\nhistogr-cnt,\nmodularity,\ntri-motifs,\nassortativity")
					->delimiter(',')->allow_extra_args(true);
	CLI11_PARSE(app, argument_count, arguments);
	
	for (const auto& arg: selected_algorithms) {
		if(arg == "node-cnt"){
			args.enable_node_count = true;
		}else if(arg == "edge-cnt") {
			args.enable_edge_count = true;
		}else if(arg == "degree-ext") {
			args.enable_degree_extr = true;
		}else if(arg == "avg-edge-len") {
			args.enable_average_edge_length = true;
		}else if(arg == "avg-apsp") {
			args.enable_all_pairs_shortest_path = true;
		}else if(arg == "avg-cluster") {
			args.enable_clustering_coefficient = true;
		}else if(arg == "avg-centrality") {
			args.enable_betweenness_centrality_average = true;
		}else if(arg == "apx-centrality") {
			args.enable_betweenness_centrality_approx = true;
	  	}else if(arg == "area-connec") {
			args.enable_area_connectivity = true;
		}else if(arg == "histogr-wid") {
			args.enable_histogram_width = true;
		}else if(arg == "histogr-cnt") {
			args.enable_histogram_count = true;
		}else if(arg == "modularity") {
			args.enable_modularity = true;
		}else if(arg == "tri-motifs") {
			args.enable_network_motifs = true;
		}else if(arg == "assortativity") {
			args.enable_assortativity = true;
		}else {
			std::cout << "The argument \"" << arg << "\" is not a valid algorithm" << std::endl;
			//terminate the programm and return a failure exitcode
			return EXIT_FAILURE;
		}
	}

	MPIWrapper::init(argument_count, arguments);

	calculate_metrics(args);
	
	/* call algorithm tests: */
	//AlgorithmTests::check_graph_property(args.input_directory);
	//AlgorithmTests::test_algorithm_parallelization(args.input_directory);

	MPIWrapper::finalize();

	return 0;
}
